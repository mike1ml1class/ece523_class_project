#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

# Import our modules
import helpers as hp

def train_ssl(clf,X_tr_lab,Y_tr_lab,X_tr_unlab,X_test,Y_test,NUM_ITER,THRESHOLD,mode):

    for i in range(NUM_ITER):

        clf.fit(X_tr_lab,Y_tr_lab)
        result = clf.predict(X_test)
        #print("Accuracy after %d rounds: %.2f" % (i,accuracy))
        idxs_d = []

        if  (X_tr_unlab.size > 0):

            for j,point in enumerate(X_tr_unlab):
                prob = clf.predict_proba(point.reshape(1,-1))
                class_w = np.argmax(prob)

                if (prob[0,class_w] > THRESHOLD):
                    # add data to labeled data
                    X_tr_lab = np.vstack((X_tr_lab,point))
                    Y_tr_lab = np.hstack((Y_tr_lab,class_w))
                    idxs_d.append(j)

            # remove data from unlabed data
            X_tr_unlab = np.delete(X_tr_unlab, idxs_d, 0)
            #print("Added %d data after this round" % (len(idxs_d)))

    #print("%.2f" % (accuracy))
    Y_pred = clf.predict(X_test)

    if (mode == 1):
        accuracy, pass_fail = hp.check_accuracy(Y_pred,Y_test)
        accuracy = accuracy/100
    else:
        accuracy = None

    return accuracy,Y_pred


def analysis(x_tr,y_tr,x_te=None,y_te=None):
    mode = 1
    #print("Performing SSL Classification!")

    # Perform K-fold cross validation
    k_fold = KFold(n_splits=5)
    clf = RandomForestClassifier(n_estimators=100)
    NUM_ITER = 2
    THRESHOLD = 0.9
    scores = []
    x_tr_values = x_tr.values
    y_tr_values = y_tr.values

    for train_indices, test_indices in k_fold.split(x_tr.values):
        x_tr_kfold = x_tr_values[train_indices]
        x_te_kfold = x_tr_values[test_indices]

        y_tr_kfold = y_tr_values[train_indices]
        y_te_kfold = y_tr_values[test_indices]

        score,temp = train_ssl(clf,
            x_tr_kfold, #labeled
            y_tr_kfold, #labeled
            x_te.values,
            x_te_kfold,
            y_te_kfold,
            NUM_ITER,THRESHOLD,mode)

        scores.append(score)

    scores = np.array(scores)


    #print("Scoring Training accuracy")
    acc,temp = train_ssl(clf,
              x_tr_values,
              y_tr_values,
              x_te.values,
              x_tr_values,
              y_tr_values,
              NUM_ITER,
              THRESHOLD,mode)

    print("\n")
    print("SSL Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    test_score = 0
    if x_te is not None:
        mode = 0
        temp,yhat = train_ssl(clf,
              x_tr_values,
              y_tr_values,
              x_te.values,
              x_te.values,
              y_te.values,
              NUM_ITER,
              THRESHOLD,mode)
        test_score, notneeded = hp.check_accuracy(yhat,y_te)
    else:
        yhat = None

    data_scores = np.array([scores.mean(),scores.std(),acc,test_score])

    return yhat,data_scores