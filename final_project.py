# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import *
from sklearn.model_selection import cross_val_score
from sklearn import svm

# Load the titanic training and testing data
data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test.csv')

data_train.info()
print('_________________________________')
print('_________________________________')
data_test.info()

# Clean the data: 
# Age, Cabin, and Embarked have missing data
# Cabin has too much missing data to be useful, so that will be dropped
# PassengerID, Name, and Ticket will also be dropped
# Take average of age and replace missing age values with average
# Embared will be changed to most common occurance
# Change male and female strings to 0 or 1
# Changed point of depature strings to 0, 1, or 2. 
# This will leave us with numeric data to use in ML models

# Drop irrelevent data
data_train = data_train.drop(['Cabin'], axis=1)
data_test  =  data_test.drop(['Cabin'], axis=1)
data_train = data_train.drop(['PassengerId'], axis=1)
data_test  =  data_test.drop(['PassengerId'], axis=1)
data_train = data_train.drop(['Ticket'], axis=1)
data_test  =  data_test.drop(['Ticket'], axis=1)
data_train = data_train.drop(['Name'], axis=1)
data_test  =  data_test.drop(['Name'], axis=1)

data_train.info()
combined_data = [data_train,data_test]

# Further clean the data 
most_common_port = data_train.Embarked.dropna().mode()[0]
average_age      = data_train.Age.dropna().mean()
for dataset in combined_data:
    dataset['Age']      = dataset['Age'].fillna(average_age)
    dataset['Sex']      = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].fillna(most_common_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


X_train = data_train.drop("Survived", axis=1)
Y_train = data_train["Survived"]

# Loop through classifiers and perform k-fold cross validation
classifiers = [svm.SVC(kernel='linear', C=1)]
class_names = ['SVM']

print("Classifier k-fold score")
for i,clf in enumerate(classifiers):
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    print("%s %.2f" % (class_names[i],scores.mean()*100))
    
    # Train with all the data
    clf.fit(X_train,Y_train)
    
    # Predict the Results with actual test data and generate CSV that 
    # Kaggle needs to actually score


