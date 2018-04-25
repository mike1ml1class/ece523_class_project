#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    print("Performing Logistic Regression Classification!")

    # Create the classifier
    log_reg = LogisticRegression()

    # Train the model
    log_reg.fit(x_tr, y_tr)

    # Classify the data
    #yhat = logreg.predict(x_te)

    # Compute the training accuracy
    acc = log_reg.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(log_reg, x_tr, y_tr, cv=5)


    print("\n")
    print("Logistic Regression Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
