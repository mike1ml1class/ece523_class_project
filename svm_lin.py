#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None,y_te=None):
    #print("Performing Linear SVM Classification!")

    # Create the SVM classifier
    svm = SVC(kernel='linear',C=1)

    if (0):

        # Genearate a feature matrix of polynomial combinations (Cover's Thm)
        poly = PolynomialFeatures(degree=3)

        # Map or transform training data to higher level
        x_tr = poly.fit_transform(x_tr)

        # Map or transform test data to higher level
        x_te = poly.fit_transform(x_te)

    # Train the models
    svm.fit(x_tr, y_tr)

    # Compute the training accuracy
    acc = svm.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(svm, x_tr, y_tr, cv=5)

    print("\n")
    print("Linear SVM Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    test_score = 0
    if x_te is not None:
        yhat = svm.predict(x_te)
        test_score, notneeded = hp.check_accuracy(yhat,y_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc,test_score])
    return yhat,data_scores