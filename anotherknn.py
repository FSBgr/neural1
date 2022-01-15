import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

# for reproducibility
np.random.seed(42)
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

y_train = train_df['label']
x_train = train_df.drop(labels=['label'], axis=1)
y_test = test_df['label']
x_test = test_df.drop(labels=['label'], axis=1)
# normalize
x_train = x_train / 10000  # x_train / 255.0
x_test = x_test / 10000  # x_test / 255.0
# convert to n-dimensional array
x_train = x_train.values
x_test = x_test.values

for i in range(1, 10):
    startTime = time.time()
    knn_clf = KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(x_train, y_train)
    preds = knn_clf.predict(x_train)
    trainAccuracy = accuracy_score(y_train, preds)
    finishTime = time.time()
    trainTime = finishTime - startTime
    print("Training Accuracy: ", trainAccuracy, " Training Time: ", trainTime)

    startTime = time.time()
    preds = knn_clf.predict(x_test)
    finishTime = time.time()
    testingAccuracy = accuracy_score(y_test, preds)
    testingTime = finishTime - startTime
    print("Testing Accuracy: ", testingAccuracy, " Testing Time: ", testingTime)
