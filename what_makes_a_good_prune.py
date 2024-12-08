# 1. The base model
# 1.1 Import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import resnet18

# 1.2 Dynamically set the device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.3 Get the training and testing dataset

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)

# 1.4 Create the training and testing functions

def train(dataloader, epochs, net):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for data in dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
        running_loss = 0.0

    print('Finished Training')

def evaluate(test_data, model):
    running_loss = 0
    total = 0
    correct = 0
    model.eval()
    torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for iters, data in enumerate(test_data):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (
        100 * correct / total,
        running_loss / iters,
    )

# 1.5 Creating LeNet5 Model

# 1.6 Training the network

net = resnet18(pretrained=True)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train(trainloader, 20, net)

# 1.7 Evaluate the model

acc, loss = evaluate(testloader,net)
print(f"Accuracy: {acc}, Loss: {loss}")

# 1.8 Save model

torch.save(net.state_dict(), "base.pth")

# 2 Calculating the optimal prune
# 2.1 Helper Functions

from torch.nn.utils import parameters_to_vector as Params2Vec
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt


def vectorise_model(model):
    """Convert Paramaters to Vector form."""
    return Params2Vec(model.parameters())

def cosine_similarity(base_weights, model_weights):
    """Calculate the cosine similairty between two vectors """
    return torch.nan_to_num(torch.clip(torch.dot(
        base_weights, model_weights
    ) / (
        torch.linalg.norm(base_weights)
        * torch.linalg.norm(model_weights)
    ),-1, 1),0)


def global_prune_without_masks(model, amount):
    """Global Unstructured Pruning of model."""
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for mod in model.modules():
        if hasattr(mod, "weight_orig"):
            if isinstance(mod.weight_orig, torch.nn.Parameter):
                prune.remove(mod, "weight")
        if hasattr(mod, "bias_orig"):
            if isinstance(mod.bias_orig, torch.nn.Parameter):
                prune.remove(mod, "bias")

def global_prune_with_masks(model, amount):
    """Global Unstructured Pruning of model."""
    parameters_to_prune = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                parameters_to_prune.append((mod, "weight"))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                parameters_to_prune.append((mod, "bias"))
    parameters_to_prune = tuple(parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

# 2.2 Prune the network between 0 and 100% caculating the Cosine Simaliarty at each prune

prune_rate = torch.linspace(0,1,101)
cosine_sim = []
base = resnet18(pretrained=True)
base.load_state_dict(torch.load("base.pth"))
base_vec = vectorise_model(base)
prune_net = resnet18(pretrained=True)

for p in prune_rate:
    p = float(p)
    prune_net.load_state_dict(torch.load("base.pth"))
    global_prune_without_masks(prune_net, p)
    prune_net_vec = vectorise_model(prune_net)
    cosine_sim.append(cosine_similarity(base_vec, prune_net_vec).item())

# 2.2 Calculate the point that is closet to the utopia, i.e 100% prune with a cosine simalairty of 1.

c = torch.vstack((torch.Tensor(cosine_sim), prune_rate))
d = c.T
dists = []
for i in d:
    dists.append(torch.dist(i, torch.Tensor([1, 1])))
min = torch.argmin(torch.Tensor(dists))

# 2.3 Plot Pareto Front


plt.plot(prune_rate, cosine_sim, label="LeNet Parateo Front")
plt.xlim(0,1.05)
plt.ylim(0,1.05)
plt.scatter(1,1,label="Utopia", c="red", marker="*", s=150)
plt.scatter(prune_rate[min], cosine_sim[min], color="k", marker="o", label="Optima")
plt.legend()
plt.grid()

# 2.4 Prune to amount speficied and finetune for 1 epoch

prune_net = resnet18(pretrained=True)
prune_net.load_state_dict(torch.load("base.pth"))
global_prune_with_masks(prune_net, float(prune_rate[min]))
prune_net.to(device)
prune_acc, prune_loss = evaluate(testloader,prune_net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(prune_net.parameters(), lr=0.001, momentum=0.9)
train(trainloader,1,prune_net)
finetune_acc, finetune_acc_loss =  evaluate(testloader,prune_net)

print(f"Base Accuracy:  {acc}, Base Loss: {loss:.3f}")
print(f"Prune Accuracy: {prune_acc}, Prune Loss: {prune_loss:.3f}")
print(f"Finetune Accuracy: {finetune_acc}, Finetune Loss: {finetune_acc_loss:.3f}")
print(f"Percentage Pruned {prune_rate[min]*100}")

# 3. The case of a high kurtosis of kurtoses

import scipy.stats as stats

def kurtosis_of_kurtoses(model):
    kurtosis = []
    for mod in model.modules():
        if hasattr(mod, "weight"):
            if isinstance(mod.weight, torch.nn.Parameter):
                kurtosis.append(stats.kurtosis(mod.weight.detach().numpy().flatten(), fisher=False))
        if hasattr(mod, "bias"):
            if isinstance(mod.bias, torch.nn.Parameter):
                kurtosis.append(stats.kurtosis(mod.bias.detach().numpy().flatten(),  fisher=False))
    kurtosis_kurtosis = stats.kurtosis(kurtosis, fisher=False)
    return kurtosis_kurtosis


kurtosis_of_kurtoses_model = kurtosis_of_kurtoses(base)

if kurtosis_of_kurtoses_model < torch.exp(torch.Tensor([1])):
    prune_modifier = 1/torch.log2(torch.Tensor([kurtosis_of_kurtoses_model]))
else:
    prune_modifier = 1/torch.log(torch.Tensor([kurtosis_of_kurtoses_model]))
safe_prune = prune_rate[min]*prune_modifier.item()
print(f"Kurtosis of kurtoses: {kurtosis_of_kurtoses_model:.2f}")
print(f"Percentage Prune: {safe_prune*100:.2f}")

prune_net = resnet18(pretrained=True)
prune_net.load_state_dict(torch.load("base.pth"))
global_prune_with_masks(prune_net, float(safe_prune))
prune_net.to(device)
prune_acc, prune_loss = evaluate(testloader,prune_net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(prune_net.parameters(), lr=0.001, momentum=0.9)
train(trainloader,1,prune_net)
finetune_acc, finetune_acc_loss =  evaluate(testloader,prune_net)

print(f"Base Accuracy:  {acc:.2f}, Base Loss: {loss:.3f}")
print(f"Prune Accuracy: {prune_acc:.2f}, Prune Loss: {prune_loss:.3f}")
print(f"Finetune Accuracy: {finetune_acc:.2f}, Finetune Loss: {finetune_acc_loss:.3f}")
print(f"Percentage Pruned {safe_prune.item()*100:.2f}")

