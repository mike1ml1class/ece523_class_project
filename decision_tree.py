#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    #print("Performing Decision Tree Classification!")

    # Create the classifier
    decision_tree = DecisionTreeClassifier()

    # Train the model
    decision_tree.fit(x_tr, y_tr)

    # Compute the training accuracy
    acc = decision_tree.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(decision_tree, x_tr, y_tr, cv=5)

    print("\n")
    print("Decision Tree Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    if x_te is not None:
        yhat = decision_tree.predict(x_te)
    else:
        yhat = None

    return yhat