import numpy as np
from rich.progress import track
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from models import model

# TODO: add output level shift + layer velocity approx from neurons + continous update of layer velocity

batch_size = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM)])

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=te_transforms)

train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

model = model().to(device)
if device == 'cpu':
    model.load_state_dict(torch.load('best_weights.pth', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load('best_weights.pth'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def is_excluded_layer(layer):
    return isinstance(layer, (nn.BatchNorm2d, nn.ReLU, nn.AvgPool2d))

initial_weights = {name: {param_name: param.clone().detach().cpu() for param_name, param in layer.named_parameters()} 
                   for name, layer in model.named_children() if isinstance(layer, nn.Module) and not is_excluded_layer(layer)}

def cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0).item()

def fine_tune_epoch(epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(track(trainloader, description=f"Training Epoch {epoch}")):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    print(f"[Epoch {epoch}] loss: {running_loss / len(trainloader):.4f}")

def compute_cosine_similarity():
    layer_similarity = {}
    for layer_name, layer in model.named_children():
        if isinstance(layer, nn.Module) and not is_excluded_layer(layer):
            if layer_name in initial_weights:
                param_similarities = []
                for param_name, param in layer.named_parameters():
                    pre_weight = initial_weights[layer_name][param_name]
                    post_weight = param.detach().cpu()
                    similarity = cosine_similarity(pre_weight, post_weight)
                    param_similarities.append(similarity)
                
                layer_similarity[layer_name] = np.mean(param_similarities)
    return layer_similarity

def freeze_all_but_most_changed_layer(weight_similarity):
    min_similarity_layer = min(weight_similarity, key=weight_similarity.get)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(min_similarity_layer)

    print(f"Freezing all layers except: {min_similarity_layer}")

fine_tune_epoch(0)

layer_similarity = compute_cosine_similarity()

print("\nCosine similarity between pre- and post-epoch 0 weights:")
for name, similarity in layer_similarity.items():
    print(f"{name}: {similarity:.6f}")

freeze_all_but_most_changed_layer(layer_similarity)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

for epoch in range(1, 10):
    fine_tune_epoch(epoch)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in track(testloader, description="Evaluating..."):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
