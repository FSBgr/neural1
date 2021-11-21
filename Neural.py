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
        super().__init__()  # Adding 4 fully connected layers, TODO: Change the number of layers, number of outputs etc for testing and result reports
        self.fc1 = nn.Linear(28 * 28, 64)  # 28 by 28 picture input
        self.fc2 = nn.Linear(64, 64)  # input is 64 since previous layer output is 64
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # Last layer output is 10, for the 10 single digit numbers

    def forward(self, x):
        x = F.relu(self.fc1(x))   # F.relu being the activation function, TODO: Change the functions through each layer and then compare in the end
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1) # returning softmax output of last layer

# TODO: initialize weights randomly
net = Net()
#print(net)

X = torch.rand((28, 28))
X = X.view(-1, 28*28) # flattening the tensor to fit the libraries, -1 specifies that input will be of unknown shape
#print(X)
output = net(X)
print(output)