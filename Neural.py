import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)  # Shuffling for better distribution
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

'''for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]

plt.imshow(data[0][0].view(28, 28))
plt.show()
# print(data[0][0].shape)
'''


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # Adding 4 layers, TODO: Change the number of layers, number of outputs etc for testing and result reports
        self.fc1 = nn.Linear(28 * 28, 64)  # 28 by 28 picture input
        self.fc2 = nn.Linear(64, 64)  # input is 64 since previous layer output is 64
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # Last layer output is 10, for the 10 single digit numbers


net = Net()
print(net)
