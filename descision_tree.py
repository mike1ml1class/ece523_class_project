#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    print("Performing Descision Tree Classification!")

    # Create the classifier
    decision_tree = DecisionTreeClassifier()

    # Train the model
    decision_tree.fit(x_tr, y_tr)

    # Classify the data
    #yhat = decision_tree.predict(x_tr)

    # Compute the training accuracy
    acc = decision_tree.score(x_tr, y_tr)

    print("\n")
    print("Decision Tree Accuracy = %3.4f" % (acc))
