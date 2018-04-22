# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import *

# Load the titanic training and testing data
data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test.csv')

# Clean the data: Change male and female strings to 0 or 1
data_train['Sex'] = data_train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
data_test['Sex']  = data_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

data_train.head()
data_test.head()