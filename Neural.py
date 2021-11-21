import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

train = datasets.MNIST("", train=True, download=True, transform= transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=True, download=True, transform= transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

x , y = data[0][0], data[1][0]
