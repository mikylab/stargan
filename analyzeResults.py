import pandas as pd
import numpy as np
from sklearn import svm
import csv 

def normalizeData(dataMatrix):
    distance_mean = np.mean(dataMatrix)
    distance_sd = np.std(dataMatrix)
    mean_zero = (dataMatrix-distance_mean)/distance_sd
    return mean_zero

## Train preparation
trainDistances = pd.read_csv("trainDistances00001_6.csv")
trainArray = np.asarray(trainDistances)
trainMatrix = trainArray[:, 1:8]

X_train = normalizeData(trainMatrix)
Y_train = trainArray[:, 0]


## Test preparation
testDistances = pd.read_csv("testDistances00001_6.csv")
testArray = np.asarray(testDistances)
testMatrix = testArray[:, 1:8]

X = normalizeData(testMatrix)
Y = testArray[:, 0]


#Train an SVM with a weight, then use the trained model to predict on the test set
#svm_model= svm.SVC(kernel="linear", probability = True, random_state = 316)
#svm_model = svm.SVC(kernel="rbf", gamma=0.7, C=1.0),
svm_model = svm.SVC(kernel="poly", degree=3, gamma="auto", C=1.0, probability = True)

svm_model.fit(X_train, Y_train)


predicted_values_train = svm_model.predict_proba(X_train)#[:,1]
predicted_values = svm_model.predict_proba(X)#[:,1]

np.save("predicted_train_poly6.npy", predicted_values_train)
np.save("predicted_values_poly6.npy", predicted_values)
