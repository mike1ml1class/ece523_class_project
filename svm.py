#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

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

    # Compute the CV scores
    scores_lin = cross_val_score(svm_lin, x_tr, y_tr, cv=5)
    scores_rbf = cross_val_score(svm_rbf, x_tr, y_tr, cv=5)


    print("\n")
    print("Lin SVM Accuracy = %3.4f" % (acc_lin))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores_lin.mean(), scores_lin.std() * 2))
    print("\n")
    print("RBF SVM Accuracy = %3.4f" % (acc_rbf))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores_rbf.mean(), scores_rbf.std() * 2))


    return