import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

EPOCHS = 3       # Change this value and run again
learningRate = 0.001       # Change this value and run again
neurons = 64     # Change this value and run again

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))   # Downloading data sets
test = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)  # Shuffling for better distribution
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # Adding 4 fully connected layers
        self.fc1 = nn.Linear(28 * 28, neurons)  # 28 by 28 picture input
        self.fc2 = nn.Linear(neurons, neurons)  # input is 64 since previous layer output is 64
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, 10)  # Last layer output is 10, for the 10 single digit numbers

        '''self.fc1 = nn.Linear(28 * 28, neurons)
        self.fc2 = nn.Linear(neurons, neurons) 
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, neurons)
        self.fc5 = nn.Linear(neurons, neurons)
        self.fc6 = nn.Linear(neurons, neurons)
        self.fc7 = nn.Linear(neurons, neurons)
        self.fc8 = nn.Linear(neurons, 10)'''

    def forward(self, x):
        x = F.relu(self.fc1(x))  # F.relu being the activation function,
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        '''x = F.relu(self.fc1(x))  # F.relu being the activation function,
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)'''

        '''x = F.softplus(self.fc1(x))  # F.relu being the activation function,
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = self.fc4(x)'''

        return F.log_softmax(x, dim=1)  # returning softmax output of last layer


neuralNetwork = MyNetwork()

images = torch.rand((28, 28))
images = images.view(-1, 28 * 28)  # flattening the tensor to fit the libraries, -1 specifies that input will be of unknown shape
output = neuralNetwork(images)

optimizer = optim.Adam(neuralNetwork.parameters(), learningRate)

correctTraining = 0
totalTraining = 0

# Beginning training process for set amount of epochs
start_time = time.time()
for epoch in range(EPOCHS):
    for data in trainSet:  # data is a batch of featuresets and labels
        images, labels = data
        neuralNetwork.zero_grad()   # Initializing the weights
        output = neuralNetwork(images.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == labels[idx]:  # comparing the accuracy, for every prediction we make, does it match the actual value
                correctTraining += 1
            totalTraining += 1
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
current_time = time.time()
training_time = current_time - start_time


correctTesting = 0
totalTesting = 0

# Beginning testing process
start_time = time.time()
with torch.no_grad():
    for data in testSet:
        images, labels = data
        output = neuralNetwork(images.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == labels[idx]:  # comparing the accuracy, for every prediction we make, does it match the actual value
                correctTesting += 1
                # print("Actual number value: ", labels[idx].item(), " Network's guess: ", torch.argmax(i).item())  # prints the actual number against the estimation, toggle on to see results
            # else:
                # print("Actual number value: ", labels[idx].item(), " Network's guess: ", torch.argmax(i).item()) # prints the actual number against the estimation, toggle on to see results
            totalTesting += 1
current_time = time.time()
testing_time = current_time - start_time


# Printing results
print("Training Time: ", training_time)
print("Testing Time: ", testing_time)
print("Training Accuracy: ", round(correctTraining / totalTraining, 3))
print("Testing Accuracy: ", round(correctTesting / totalTesting, 3))


'''print("My guess is: ", torch.argmax(neuralNetwork(images[0].view(-1, 28 * 28))[0]).item()) # prints the network's guess of the first image of the dataset
plt.imshow(images[0].view(28, 28))  # prints the first image of the dataset
plt.show()'''