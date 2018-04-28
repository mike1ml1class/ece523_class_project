#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Import our modules
import helpers as hp

def analysis(x_tr,y_tr,x_te=None):
    
    # Set up to perform k-fold cross validation
    k_fold = KFold(n_splits=5)
    HIDDEN = [100,100,100,100,100]
    NUM_STEPS = 1000

    feature_columns = [tf.feature_column.numeric_column("x", shape=[1, x_tr.values.shape[1]])]
    
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=HIDDEN,
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=2,
        dropout=0.1)

    scores = []
    for train_indices, test_indices in k_fold.split(x_tr.values):
        x_tr_values = x_tr.values
        y_tr_values = y_tr.values
   
        x_tr_kfold = x_tr_values[train_indices]
        x_te_kfold = x_tr_values[test_indices]
        y_tr_kfold = y_tr_values[train_indices]
        y_te_kfold = y_tr_values[test_indices]

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_tr_kfold},
            y=y_tr_kfold,
            num_epochs=None,
            batch_size=50,
            shuffle=True)
        
        classifier.train(input_fn=train_input_fn, steps=NUM_STEPS)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_te_kfold},
            y=y_te_kfold,
            num_epochs=1,
            shuffle=False)
        scores.append(classifier.evaluate(input_fn=train_input_fn)["accuracy"])
        
    scores = np.array(scores)
    # Set up to get training accuracy
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_tr.values},
        y=y_tr.values,
        num_epochs=None,
        batch_size=50,
        shuffle=True) 
        
    classifier.train(input_fn=train_input_fn, steps=NUM_STEPS)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_tr.values},
        y=y_tr.values,
        num_epochs=1,
        shuffle=False)
        
    acc = classifier.evaluate(input_fn=train_input_fn)["accuracy"]

    print("\n")
    print("DNN Accuracy = %3.4f" % (acc))
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Classify the data
    if x_te is not None:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_te.values},
            num_epochs=1,
            shuffle=False)
        
        predictions = classifier.predict(input_fn=train_input_fn)
        y_pred = []
        for i in predictions:
            y_pred.append(int(i['classes']))
        yhat = np.array(y_pred)
    else:
        yhat = None
    
    data_scores = np.array([scores.mean(),scores.std(),acc])
    return yhat,data_scores