import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
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

# Creating the model and fitting it
pca = PCA(n_components=5)
pca.fit(trainingImages[0][0])
transformedTrainingImages = pca.transform(trainingImages[0][0])
transformedTestingImages = pca.transform(testingImages[0][0])

transformedTrainingImages = pca.inverse_transform(transformedTrainingImages)
transformedTestingImages = pca.inverse_transform(transformedTestingImages)


print("Training success:" , (1-mean_squared_error(trainingImages[0][0], transformedTrainingImages))*100)
print("Testing success:" , (1-mean_squared_error(testingImages[0][0], transformedTestingImages))*100)