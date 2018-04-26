#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    #print("Performing MLP Classification!")

    # Create the classifier
    clf = MLPClassifier(solver='sgd',alpha=1e2,hidden_layer_sizes=(100,100),
                        random_state=1,max_iter = 400,momentum=0.9,
                        learning_rate_init=0.2)

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
    if x_te is not None:
        yhat = clf.predict(x_te)
    else:
        yhat = None

    return yhat
