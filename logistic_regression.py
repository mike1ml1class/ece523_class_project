#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None,y_te=None):
    #print("Performing Logistic Regression Classification!")

    # Create the classifier
    log_reg = LogisticRegression()

    if (1):

        # Genearate a feature matrix of polynomial combinations (Cover's Thm)
        poly = PolynomialFeatures(degree=3)

        # Map or transform training data to higher level
        x_tr = poly.fit_transform(x_tr)

        # Map or transform test data to higher level
        x_te = poly.fit_transform(x_te)

    # Train the model
    log_reg.fit(x_tr, y_tr)

     # Compute the training accuracy
    acc = log_reg.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(log_reg, x_tr, y_tr, cv=5)

    print("\n")
    print("Logistic Regression Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    if x_te is not None:
        yhat = log_reg.predict(x_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc])
    return yhat,data_scores