# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import *

# Load the titanic training and testing data
data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test.csv')

combined_data = [data_train,data_test]

# Clean the data: 
# - Change male and female strings to 0 or 1
# - Changed point of depature strings to 0, 1, or 2. Fill in missing
#   data with most common occurance

most_common_port = data_train.Embarked.dropna().mode()[0]
for dataset in combined_data:
    dataset['Sex']      = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].fillna(most_common_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



data_train.head()
data_test.head()