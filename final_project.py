#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project

# Import necessary libraries
import pandas as pd

# Import our modules
import titanic
import knn as knn
import logistic_regression as lr
import random_forest as rf
import svm_lin as svm_lin
import svm_rbf as svm_rbf
import decision_tree as dt
import pca as pca
import dnn as dnn
import adaboost as ada
import bagging as bag
import semissl as semissl
import mlp_sk  as mlp_sk

# Setup and Config
VISUALIZE     = True
PERFORM_PCA   = True
PERFORM_CLASS = True
GEN_OUTPUT    = True
ALT_DATA      = True

# Load the titanic training and testing data
data_train = pd.read_csv('train.csv')
#data_test  = pd.read_csv('test.csv')
data_test  = pd.read_csv('test_w_survive_data.csv')

# Convert categorical data to numerical and fill in missing values
data = titanic.fix_fill_convert(data_train,data_test,ALT_DATA)

# Extract the fixed data frames
df_tr = data[0]                  # Training data
df_te = data[1]                  # Testing data

# Create features and labels
x_tr = df_tr.drop("Survived", axis=1)
y_tr = df_tr["Survived"]
x_te = df_te.drop("Survived", axis=1)
y_te = df_te["Survived"]

if VISUALIZE:
    # Visualize data and correlations to survival
    titanic.visualize_data(data_train,data_test)

if PERFORM_PCA:
    # Create Reduced Feature Data using PCA
    # -Using three components to reduce features.  Will plot
    #  each axis against the others to see if there is any
    #  good separation.
    #
    # -Looking at plot there does not appear
    #  to be. Indicates data is not linearly seperable and more
    #  sophisticated techniques will be needed to get a decent
    #  score
    pca.analysis(x_tr.values,y_tr.values)

if PERFORM_CLASS:
    # Classifier dictionary
    clf_dict = {'LinSVM'             : svm_lin ,
                'RbfSVM'             : svm_rbf ,
                'LogisticRegression' : lr      ,
                'RandomForest'       : rf      ,
                'KNN'                : knn     ,
                'DecisionTree'       : dt      ,
                'AdaBoost'           : ada     ,
                'Bagging'            : bag     ,
                'MLP_sk'             : mlp_sk  ,
                'DNN'                : dnn     ,
                'SemiSupervised'     : semissl}

    # Empty results dictionary
    clf_results = {};

    # Loop over the classifiers
    for key,clf in clf_dict.items():

        # Perform classification
        yhat,score_data = clf.analysis(x_tr,y_tr,x_te,y_te)

        # Store results in dictionary
        clf_results[key] = [yhat,score_data]

if GEN_OUTPUT:

    print("\n")

    for key,clf_result in clf_results.items():

        print('Generating %s Output...' % (key))

        # Create filename
        fn = key + '_submission.csv'
        yhat = clf_result[0]
        # Generate the submission file
        titanic.create_submission(data_test['PassengerId'],yhat,fn)

    print('Generating Results Table...')
    print("\n")
    print("Classifier\tTraining Score\t k-fold Score\t(+/-) Test Score")
    for key,clf_result in clf_results.items():
        raw = clf_result[1]
        print("%s & %3.4f & %0.2f & %0.2f & %0.2f\\\\" % (key,raw[2]*100,raw[0]*100,raw[1]*100,raw[3]))
        
        
        