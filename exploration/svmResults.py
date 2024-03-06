import pandas as pd
import numpy as np
from sklearn import svm
import csv 
import sys
import os
import argparse

def normalizeData(dataMatrix):
    distance_mean = np.mean(dataMatrix)
    distance_sd = np.std(dataMatrix)
    mean_zero = (dataMatrix-distance_mean)/distance_sd
    return mean_zero
    
    

def main(config):
    #file_path = '/home/mikylab/github/stargan/experiments/stargan_identity.bel01/test_distances.csv'
    #print(file_path)
    trainDistances = pd.read_csv(config.dist_dir)
    trainArray = np.asarray(trainDistances)
    trainMatrix = trainArray[:, 1:3]

    X_train = normalizeData(trainMatrix)
    Y_train = trainArray[:, 0]



    #Train an SVM with a weight, then use the trained model to predict on the test set
    if config.svm == 'linear':
        svm_model= svm.SVC(kernel="linear", probability = True, random_state = 316)
    elif config.svm == 'rbf':
        svm_model = svm.SVC(kernel="rbf", gamma=0.7, C=1.0),
    else:
        svm_model = svm.SVC(kernel="poly", degree=3, gamma="auto", C=1.0, probability = True)

    svm_model.fit(X_train, Y_train)
    predicted_values_train = svm_model.predict_proba(X_train)
    np.save(config.result_dir, predicted_values_train)
    #np.save('/home/mikylab/github/stargan/experiments/stargan_identity.bel01/predicted_beltest_lin.npy', predicted_values_train)

    class_predictions_test = np.argmax(predicted_values_train, axis =1 )
    correct_test = np.sum(Y_train == class_predictions_test)
    accuracy = correct_test/class_predictions_test.shape[0]
    print(accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Model configuration
    parser.add_argument('--dist_dir', type = str, help = "Path to the file with image translation distances")
    parser.add_argument('--svm', type = str, default = 'linear', help = "Type of SVM to use")
    parser.add_argument('--result_dir', type = str, default = 'stargan/results')
    config = parser.parse_args()
    main(config)
