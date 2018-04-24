#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    print("Performing SVM Classification!")

    # Create the SVM classifiers
    svm_lin = SVC(kernel='linear')
    svm_rbf = SVC(kernel='rbf',C=0.25)

    # Train the models
    svm_lin.fit(x_tr, y_tr)
    svm_rbf.fit(x_tr, y_tr)

    # Classify the data
    #yhat_lin = svm_lin.predict(x_te)
    #yhat_rbf = svm_rbf.predict(x_te)

    # Compute the training accuracy
    acc_lin = svm_lin.score(x_tr, y_tr)
    acc_rbf = svm_rbf.score(x_tr, y_tr)

    print("\n")
    print("Lin SVM Accuracy = %3.4f" % (acc_lin))
    print("RBF SVM Accuracy = %3.4f" % (acc_rbf))

    return