#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

# Import our modules
import helpers as hp
import titanic
import knn as knn
import logistic_regression as lr
import random_forest as rf
import svm as svm
import decision_tree as dt
import pca as pca
import dnn as dnn



# Setup and Config
GEN_OUTPUT    = True
PERFORM_CLASS = True
NEURAL_NET    = True
VISUALIZE     = True

# Load the titanic training and testing data
data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test.csv')

# Visualize data and correlations to survival
if VISUALIZE:
    titanic.visualize_data(data_train,data_test)

# Convert categorical data to numerical and fill in missing values
data = titanic.fix_fill_convert(data_train,data_test)

# Extract the fixed data frames
df_tr = data[0]                  # Training data
df_te = data[1]                  # Testing data


if PERFORM_CLASS:
    x_tr = df_tr.drop("Survived", axis=1)
    y_tr = df_tr["Survived"]
    x_te = df_te

    # Loop through classifiers and perform k-fold cross validation
    #classifiers = [svm.SVC(kernel='linear', C=1)]
    
    
    classifiers = [svm.analysis,
                   lr.analysis,
                   rf.analysis,
                   knn.analysis,
                   dt.analysis]
                   
    class_names = ['SVM','LogisticRegression','RandomForest','KNN','DecsionTree']

    # Create Reduced Feature Data using PCA
    # -Using three components to reduce features.  Will plot
    #  each axis against the others to see if there is any
    #  good separation.
    #
    # -Looking at plot there does not appear
    #  to be. Indicates data is not linearly seperable and more
    #  sophisticated techniques will be needed to get a decent
    #  score

    #pca.analysis(x_tr,y_tr)

    classifier_scores = []
    print("Classifier k-fold score")
    for i,my_clf in enumerate(classifiers):
        #scores = cross_val_score(clf, x_tr, y_tr, cv=5)
        #print("%s %.2f" % (class_names[i],scores.mean()*100))
        #classifier_scores.append(scores.mean()*100)
        
        my_clf(x_tr,y_tr)

        # Train with all the data?
        #clf.fit(x_tr,y_tr)

        # Predict the Results with actual test data and generate CSV that
        # Kaggle needs to actually score
        #y_pred = clf.predict(x_te)

    if GEN_OUTPUT:
        print('Generating Output...')
        #file_str = './output/' + class_names[i] + '_' + 'submission.csv'
        #titanic.create_submission(data_test,y_pred,file_str)

if NEURAL_NET:
    dnn.analysis()
