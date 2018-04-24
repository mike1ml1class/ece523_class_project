#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    print("Performing KNN Classification!")

    # Create the classifier
    knn = KNeighborsClassifier(n_neighbors = 3)

    # Train the model
    knn.fit(x_tr, y_tr)

    # Classify the test data
    #yhat = knn.predict(x_te)

    # Compute the training accuracy
    acc_knn = knn.score(x_tr, y_tr)


    print("\n")
    print("KNN Accuracy = %3.4f" % (acc_knn))


