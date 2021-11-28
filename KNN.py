import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import time

MNISTdata = load_digits()
trainSet, testData, trainLabels, testLabel = train_test_split(np.array(MNISTdata.data), MNISTdata.target, test_size=0.25, random_state=42) # exporting data

trainSet, dataValues, trainLabels, labelValues = train_test_split(trainSet, trainLabels, test_size=0.1, random_state=84)

starting_time = time.time()
exampleModel = KNeighborsClassifier(n_neighbors=5) # running KNN to train the model
exampleModel.fit(trainSet, trainLabels)
predictions = exampleModel.predict(testData)        # predicting values based on trained model via KNN
current_time = time.time()

print("KNN finished in: ", current_time - starting_time, " seconds") # printing results
print(classification_report(testLabel, predictions))
