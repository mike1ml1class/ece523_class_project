import pandas as pd
import numpy as np


data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test2.csv')

data_test

data_survivors = pd.read_csv('survivors.csv')
data_victims   = pd.read_csv('victims.csv')


data = [data_survivors,data_victims]

for idx,df in enumerate(data):
    df['FullName'] = df['LastName']+ ", " + df['FirstName']
    
    for temp in df.iterrows():
        row = temp[1]
        #print(row['LastName'])
        #name = row
        data_test[data_test['Name'].str.contains(row['LastName'])]
        data_train[data_train['Name'].str.contains(row['LastName'])]

#df['Title'] = df['Title'].replace('Mlle', 'Miss')
#df['Title'] = df['Title'].replace('Ms', 'Miss')
#df['Title'] = df['Title'].replace('Mme', 'Mrs')