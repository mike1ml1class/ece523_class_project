#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    print("Performing Random Forest Classification!")

    # Create the classifier
    random_forest = RandomForestClassifier(n_estimators=100)

    # Train the model
    random_forest.fit(x_tr, y_tr)

    # Classify the data
    #yhat = random_forest.predict(x_te)

    # Compute the training accuracy
    acc = random_forest.score(x_tr, y_tr)

    print("\n")
    print("Random Forest Accuracy = %3.4f" % (acc))