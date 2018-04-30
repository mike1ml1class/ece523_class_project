#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None,y_te=None):
    #print("Performing MLP Classification!")

    # Create the classifier
    clf = MLPClassifier(solver='adam',alpha=1e5,hidden_layer_sizes=(100,100),
                        random_state=1,max_iter = 500,momentum=0.9,
                        learning_rate_init=0.002)

    # Create a scaler to scale the features (mean=0, var=1)
    scaler = StandardScaler()

    # Fit
    scaler.fit(x_tr)

    # Scale
    x_tr = scaler.transform(x_tr)
    x_te = scaler.transform(x_te)

    # Train the model
    clf.fit(x_tr, y_tr)

    # Compute the training accuracy
    acc = clf.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(clf, x_tr, y_tr, cv=5)

    print("\n")
    print("MLP_SK Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    test_score = 0
    if x_te is not None:
        yhat = clf.predict(x_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc,test_score])
    return yhat,data_scores
