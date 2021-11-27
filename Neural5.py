import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)  # Shuffling for better distribution
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()  # Adding 4 fully connected layers
        self.fc1 = nn.Linear(28 * 28, 64)  # 28 by 28 picture input
        self.fc2 = nn.Linear(64, 64)  # input is 64 since previous layer output is 64
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # Last layer output is 10, for the 10 single digit numbers

    def forward(self, x):
        x = F.relu(self.fc1(x))  # F.relu being the activation function,
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)  # returning softmax output of last layer


net = Net()

X = torch.rand((28, 28))
X = X.view(-1, 28 * 28)  # flattening the tensor to fit the libraries, -1 specifies that input will be of unknown shape
output = net(X)

optimizer = optim.Adam(net.parameters(), lr=0.001)

# EPOCHS = 6
EPOCHS = 2

correctTR = 0
totalTR = 0

start_time = time.time()
for epoch in range(EPOCHS):
    for data in trainset:  # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:  # comparing the accuracy, for every prediction we make, does it match the actual value
                correctTR += 1
            totalTR += 1
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
current_time = time.time()
training_time = current_time - start_time


correctTE = 0
totalTE = 0

start_time = time.time()
with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:  # comparing the accuracy, for every prediction we make, does it match the actual value
                correctTE += 1
            totalTE += 1
current_time = time.time()
testing_time = current_time - start_time

print("Training Time: ", training_time)
print("Testing Time: ", testing_time)
print("Training Accuracy: ", round(correctTR / totalTR, 3)) # printing results
print("Testing Accuracy: ", round(correctTE / totalTE, 3)) # printing results
plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(net(X[0].view(-1, 28 * 28))[0])) # prints the network's guess of the previously shown number on the graph
