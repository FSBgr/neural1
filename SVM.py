import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()  # loading the 0-9 digits dataset

classifier = svm.SVC(gamma=0.0001, C=100)  # TODO: tweak the gamma

trainingVectors, targetValues = digits.data[:-719], digits.target[:-719]  # dataset contains 1797 digits, so I use 60% of it as training, meaning up until the 719th element
classifier.fit(trainingVectors, targetValues)

for i in range(-718 , -1):  # using the rest 40% of the set as testing set. This means starting from the 718th to last element until the last element
    print("Prediction: ", classifier.predict(digits.data[i].reshape(1, -1)), " Actual Value: ", digits.target[i])
    # plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    # plt.show()

testSamples = digits.data[-719:] # taking the last 719 elements of the data to use as testing set
trueLabels = digits.target[-719:]
# print("The training accuracy is: ", clf.score(trainingVectors, targetValues)) # not correct yet
print("The prediction/testing accuracy is: ", classifier.score(testSamples, trueLabels))

'''trainingGuesses = 0
for i in range(-1797, -1078):
    if classifier.predict(digits.data[i].reshape(1, -1)) == digits.target[i]:
        trainingGuesses += 1
print(trainingGuesses)'''

#TODO: figure out a way to measure the training accuracy
