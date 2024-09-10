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

# NEq applied layer-wise, with layer-wise velocity approximated by random-sampling of intra-layer neurons
# TODO: add continuous update of layer velocity

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

def add_noise(inputs, noise_level=0.1):
    noise = torch.randn_like(inputs) * noise_level
    noisy_inputs = inputs + noise
    return torch.clamp(noisy_inputs, 0, 1)

def flip(labels):
    flipped_labels = 9 - labels
    return flipped_labels

def fine_tune_epoch(epoch, cifar_f=False, cifar_c=False, noise_level=0.1):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(track(trainloader, description=f"Training Epoch {epoch}")):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        if cifar_c:
            inputs = add_noise(inputs, noise_level=noise_level)

        if cifar_f:
            labels = flip(labels)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()

    print(f"[Epoch {epoch}] loss: {running_loss / len(trainloader):.4f}")

def select_neurons(layer, fraction=0.1):
    total_neurons = layer.weight.shape[0]
    selected_neurons = int(total_neurons * fraction)
    indices = torch.randperm(total_neurons)[:selected_neurons]
    return indices

def freeze_all_except(weight_similarity):
    min_similarity_layer = min(weight_similarity, key=weight_similarity.get)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(min_similarity_layer)
    print(f"freezing all layers except: {min_similarity_layer}")

def freeze_all_but(layer, selected_indices):
    layer.weight.grad = torch.zeros_like(layer.weight)
    layer.weight.grad[selected_indices] = layer.weight.grad[selected_indices].clone()
    layer.weight.data[selected_indices] -= optimizer.defaults['lr'] * layer.weight.grad[selected_indices]

def recursively_freeze(module, fraction=0.1):
    for child_name, child in module.named_children():
        if is_excluded_layer(child):
            continue
        elif hasattr(child, 'weight') and isinstance(child.weight, torch.nn.Parameter):
            child.weight.requires_grad = True
            selected_indices = select_neurons(child, fraction)
            freeze_all_but(child, selected_indices)
        elif isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)) or hasattr(child, 'children'):
            recursively_freeze(child, fraction)

print("\nevaluating ILS...")

for layer_name, layer in model.named_children():
    if isinstance(layer, nn.Module) and not is_excluded_layer(layer):
        recursively_freeze(layer, fraction=0.1)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

fine_tune_epoch(0, cifar_c=True)

layer_similarity = compute_cosine_similarity()

print("\ncosine similarity between pre- and post-epoch 0 weights:")
for name, similarity in layer_similarity.items():
    print(f"{name}: {similarity:.6f}")

freeze_all_except(layer_similarity)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

for epoch in range(1, 10):
    fine_tune_epoch(epoch, cifar_c=True)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in track(testloader, description="evaluating..."):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = add_noise(inputs, noise_level=0.1)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'acc: {100 * correct / total:.2f}%')

print("\n\nevaluating OLS...")

initial_weights = {name: {param_name: param.clone().detach().cpu() for param_name, param in layer.named_parameters()} 
                   for name, layer in model.named_children() if isinstance(layer, nn.Module) and not is_excluded_layer(layer)}

for layer_name, layer in model.named_children():
    if isinstance(layer, nn.Module) and not is_excluded_layer(layer):
        recursively_freeze(layer, fraction=0.1)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

fine_tune_epoch(0, flip_labels_flag=True)

layer_similarity_flipped = compute_cosine_similarity()

print("\ncosine similarity between pre- and post-epoch 0 weights:")
for name, similarity in layer_similarity_flipped.items():
    print(f"{name}: {similarity:.6f}")

freeze_all_except(layer_similarity_flipped)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

for epoch in range(1, 10):
    fine_tune_epoch(epoch, flip_labels_flag=True)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in track(testloader, description="evaluating..."):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        labels = flip_labels(labels)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'acc: {100 * correct / total:.2f}%')
