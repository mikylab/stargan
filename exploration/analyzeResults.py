import pandas as pd
import numpy as np
from sklearn import svm
import csv 
import sys

def normalizeData(dataMatrix):
    distance_mean = np.mean(dataMatrix)
    distance_sd = np.std(dataMatrix)
    mean_zero = (dataMatrix-distance_mean)/distance_sd
    return mean_zero


def main():
    if len(sys.argv) != 2:
        print("Missing path.")
    file_name = sys.argv[1]
    
    ## Train preparation
    #trainDistances = pd.read_csv('/home/mikylab/stargan-identity/stargan_identity.bel0/results/trainDistances_bel60.csv')
    file_path = '/home/mikylab/github/stargan/experiments/stargan_identity.bel01/test_distances.csv'
    print(file_path)
    trainDistances = pd.read_csv(file_path)
    trainArray = np.asarray(trainDistances)
    trainMatrix = trainArray[:, 1:3]

    X_train = normalizeData(trainMatrix)
    Y_train = trainArray[:, 0]


## Test preparation
# testDistances = pd.read_csv('/home/mikylab/stargan-identity/stargan_identity.hair/results/testDistance720.csv')
# testArray = np.asarray(testDistances)
# testMatrix = testArray[:, 1:8]

# X = normalizeData(testMatrix)
# Y = testArray[:, 0]


    #Train an SVM with a weight, then use the trained model to predict on the test set
    svm_model= svm.SVC(kernel="linear", probability = True, random_state = 316)
    #svm_model = svm.SVC(kernel="rbf", gamma=0.7, C=1.0),
    #svm_model = svm.SVC(kernel="poly", degree=3, gamma="auto", C=1.0, probability = True)

    svm_model.fit(X_train, Y_train)


    predicted_values_train = svm_model.predict_proba(X_train)#[:,1]
    #predicted_values = svm_model.predict_proba(X)#[:,1]

    np.save('/home/mikylab/github/stargan/experiments/stargan_identity.bel01/predicted_beltest_lin.npy', predicted_values_train)
    #np.save("/home/mikylab/stargan-identity/stargan_identity.bel0/results/predicted_beltrain60_lin.npy", predicted_values_train)
#np.save("/home/mikylab/stargan-identity/stargan_identity.hair/results/predicted_test_poly.npy", predicted_values)

if __name__ == "__main__":
        main()
