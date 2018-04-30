#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None,y_te=None):
    #print("Performing Random Forest Classification!")

    # Create the classifier
    random_forest = RandomForestClassifier(n_estimators=100)

    # Train the model
    random_forest.fit(x_tr, y_tr)

    # Compute the training accuracy
    acc = random_forest.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(random_forest, x_tr, y_tr, cv=5)

    print("\n")
    print("Random Forest Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    if x_te is not None:
        yhat = random_forest.predict(x_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc])
    return yhat,data_scores
