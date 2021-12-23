import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import time

digits = datasets.load_digits()  # loading the 0-9 digits dataset
trainingSize = round(len(digits.data) * 0.6)  # declaring the size of the training set and the test set
testingSize = round(len(digits.data) * 0.4)


classifier = svm.SVC(gamma=0.001, C=100)  # also try for gamma=0.1 and gamma=0.0001
# classifier = svm.SVC(kernel="rbf")  # also try kernel="poly", kernel="rbf", kernel="linear"

startTime = time.time()
trainingSamples, trainingLabels = digits.data[:trainingSize], digits.target[:trainingSize]
classifier.fit(trainingSamples, trainingLabels)
finishTime = time.time()
print("Training was completed in: ", finishTime - startTime, " seconds.")

# toggle on the following for loop to print the images and compare them with the SVM's guesses

# for i in range(trainingSize):
#   print("Prediction: ", classifier.predict(digits.data[i].reshape(1, -1)), " Actual Value: ", digits.target[i])
#   plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")

testSamples = digits.data[trainingSize + 1: len(digits.data)]  # using the rest 40% of the set as testing set
testLabels = digits.target[trainingSize + 1: len(digits.data)]
print("The training accuracy is: ", classifier.score(trainingSamples, trainingLabels))
startTime = time.time()
print("The prediction/testing accuracy is: ", classifier.score(testSamples, testLabels))
finishTime = time.time()
print("Testing finished in: ", finishTime - startTime, " seconds.")
