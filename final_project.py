# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import *

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
# Take average of age and replace missing age values with average
# Embared will be changed to most common occurance
# Change male and female strings to 0 or 1
# Changed point of depature strings to 0, 1, or 2. 

data_train = data_train.drop(['Cabin'], axis=1)
data_test  = data_test.drop(['Cabin'], axis=1)
data_train.info()
combined_data = [data_train,data_test]

most_common_port = data_train.Embarked.dropna().mode()[0]
average_age      = data_train.Age.dropna().mean()
for dataset in combined_data:
    dataset['Age']      = dataset['Age'].fillna(average_age)
    dataset['Sex']      = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].fillna(most_common_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data_train.head()
data_test.head()