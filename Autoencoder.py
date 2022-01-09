import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# Loading data
transform = transforms.ToTensor()
trainingData = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainingData_Loader = torch.utils.data.DataLoader(dataset=trainingData, batch_size=64, shuffle=True)
trainingDataDataiter = iter(trainingData_Loader)
trainingImages, trainingLabels = trainingDataDataiter.next()

testingData = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testingData_Loader = torch.utils.data.DataLoader(dataset=testingData, batch_size=64, shuffle=True)
testingDataDataiter = iter(testingData_Loader)
testingImages, testingLabels = testingDataDataiter.next()


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()  # try pass if it doesn't work
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # N, 784 -> N, 128, 28*28 because of the image size
            nn.ReLU(),
            nn.Linear(128, 64),  # nn.Linear(128,128)
            nn.ReLU(),
            nn.Linear(64, 12),  # nn.Linear(128,128)
            nn.ReLU(),
            nn.Linear(12, 3)  # nn.Linear(128,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),  # nn.Linear(3,128)     # N, 784 -> N, 128
            nn.ReLU(),
            nn.Linear(12, 64),  # nn.Linear(128,128)
            nn.ReLU(),
            nn.Linear(64, 128),  # nn.Linear(128,128)
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # need this in order to produce image sizes between [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


myAutoencoder = Autoencoder()
lossFunction = nn.MSELoss()
opt = torch.optim.Adam(myAutoencoder.parameters(), lr=0.001, weight_decay=0.00001)

EPOCHS = 3

# training phase

output = []
start_time = time.time()
for epoch in range(EPOCHS):
    for (img, _) in trainingData_Loader:
        img = img.reshape(-1, 28 * 28)
        reconstruct = myAutoencoder(img)  # get the reconstructed image
        loss = lossFunction(reconstruct, img)  # calculating min square error

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}')
    output.append((epoch, img, reconstruct))

finish_time = time.time()
training_time = finish_time - start_time

# printing the images

plt.figure(figsize=(9, 2))
plt.gray()
realImages = output[0][1].detach().numpy()
reconstructedImages = output[0][2].detach().numpy()

for i, item in enumerate(realImages):
    if i >= 9: break
    plt.subplot(2, 9, i + 1)
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
plt.show()

for i, item in enumerate(reconstructedImages):
    if i >= 9: break
    plt.subplot(2, 9, i + 1)
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
plt.show()

# testing phase

output = []
start_time = time.time()
for (img, _) in testingData_Loader:  # data_Loader:
    img = img.reshape(-1, 28 * 28)
    reconstruct = myAutoencoder(img)  # get the reconstructed image
    loss = lossFunction(reconstruct, img)  # calculating min square error

    opt.zero_grad()
    loss.backward()
    opt.step()
print("\n\nTESTING RESULTS\n\n")
print(f'Loss: {loss.item():.3f}')
output.append((img, reconstruct))

finish_time = time.time()
testingTime = finish_time - start_time

plt.figure(figsize=(9, 2))
plt.gray()
realImages = output[0][0].detach().numpy()
reconstructedImages = output[0][1].detach().numpy()
for i, item in enumerate(realImages):
    if i >= 9: break
    plt.subplot(2, 9, i + 1)
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
# plt.show()

for i, item in enumerate(reconstructedImages):
    if i >= 9: break
    plt.subplot(2, 9, i + 1)
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
# plt.show()

print("training: ", training_time, "    testing: ", testingTime)
