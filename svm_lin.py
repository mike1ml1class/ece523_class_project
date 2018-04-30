#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None,y_te=None):
    #print("Performing Linear SVM Classification!")

    # Create the SVM classifier
    svm = SVC(kernel='linear',C=0.25)

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
    if x_te is not None:
        yhat = svm.predict(x_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc])
    return yhat,data_scores