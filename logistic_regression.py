#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None,y_te=None):
    #print("Performing Logistic Regression Classification!")

    # Create the classifier
    clf = LogisticRegression(C=0.7)

    if (1):

        # Genearate a feature matrix of polynomial combinations (Cover's Thm)
        poly = PolynomialFeatures(degree=3)

        # Map or transform training data to higher level
        x_tr = poly.fit_transform(x_tr)

        # Map or transform test data to higher level
        x_te = poly.fit_transform(x_te)

    # Train the model
    clf.fit(x_tr, y_tr)

     # Compute the training accuracy
    acc = clf.score(x_tr, y_tr)

    # Compute the CV scores
    scores = cross_val_score(clf, x_tr, y_tr, cv=5)

    print("\n")
    print("Logistic Regression Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    test_score = 0
    if x_te is not None:
        yhat = clf.predict(x_te)
        test_score, notneeded = hp.check_accuracy(yhat,y_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc,test_score])
    return yhat,data_scores


def coefs(x_tr,y_tr,x_te=None,y_te=None):


    # Create the classifier
    clf = LogisticRegression()

    # Train the model
    clf.fit(x_tr, y_tr)

    # Compute the training accuracy
    acc = clf.score(x_tr, y_tr)

    # Get the correlation coefs
    coeff_df = pd.DataFrame(x_tr.columns)
    coeff_df.columns = ['Feature']
    coeff_df["Correlation"] = pd.Series(clf.coef_[0])
    coeff_df["AbsCorrelation"] = pd.Series(np.abs(clf.coef_[0]))

    print("\n")
    print(coeff_df.sort_values(by='Correlation', ascending=False))
    print("\n")
    print(coeff_df.sort_values(by='AbsCorrelation', ascending=False))


    f,ax = plt.subplots(figsize=(8,8))
    ax.set_title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(x_tr.astype(float).corr(method='pearson'),linewidths=0.1,vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
    f.savefig('pearson_corr.png')

    f,ax = plt.subplots(figsize=(8,8))
    ax.set_title('Spearman Correlation of Features', y=1.05, size=15)
    sns.heatmap(x_tr.astype(float).corr(method='spearman'),linewidths=0.1,vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
    f.savefig('spearman_corr.png')
