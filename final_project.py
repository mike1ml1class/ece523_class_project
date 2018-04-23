# Final project scripts
# cd C:\Users\Michael\Desktop\ECE523\project\ece523_class_project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import *
from sklearn.model_selection import cross_val_score
from sklearn import svm
import sys

GEN_OUTPUT = False

# Load the titanic training and testing data
data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test.csv')

data_train_orig = data_train
data_test_orig  = data_test

print('\nTraining Data:')
print( data_train.info() )

print('\nTesting Data:')
print( data_test.info() )

print('\nDescribe Training:')
print( data_train.describe() )



# Clean the data:
# Age, Cabin, and Embarked have missing data
# Cabin has too much missing data to be useful, so that will be dropped
# PassengerID, Name, and Ticket will also be dropped
# Take average of age and replace missing age values with average
# Embared will be changed to most common occurance
# Change male and female strings to 0 or 1
# Changed point of depature strings to 0, 1, or 2.
# This will leave us with numeric data to use in ML models
# Test data has a missing value for Fare, this will be replaced
# by the average fare

# Do a bit of analysis by isolating features
print('\n')
out = data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(out)

print('\n')
out = data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(out)

print('\n')
out = data_train[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(out)

print('\n')
out = data_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(out)

print('\n')
out = data_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(out)



f,ax = plt.subplots(1,2,figsize=(8,4))
#ax = data_train.hist(column='Age',by='Survived',sharey=True,xrot=45,bins=20,ec='k')
data_train.hist(ax=ax,column='Age',by='Survived',rot=0,bins=20,ec='k')
ax[0].set_xlabel('Age')
ax[1].set_xlabel('Age')
ax[0].grid(True)
ax[1].grid(True)
ax[0].set_title('Survived = 0')
ax[1].set_title('Survived = 1')
plt.tight_layout()
f.savefig('Age_hist')

f,ax = plt.subplots(3,2,figsize=(8,8))
#data_train.hist(ax=ax,column='Age',by=['Pclass','Survived'],sharey=True,sharex=True,xrot=45,bins=20,ec='k')
data_train.hist(ax=ax,column='Age',by=['Pclass','Survived'],xrot=45,bins=20,ec='k')
ax[2,0].set_xlabel('Age')
ax[0,0].set_title('Pclass = 1,Survived = 0')
ax[0,1].set_title('Pclass = 1,Survived = 1')
ax[1,0].set_title('Pclass = 2,Survived = 0')
ax[1,1].set_title('Pclass = 2,Survived = 1')
ax[2,0].set_title('Pclass = 3,Survived = 0')
ax[2,1].set_title('Pclass = 3,Survived = 1')
plt.tight_layout()
f.savefig('Pclass_hist')


'''

# Drop irrelevent features that should not effect survival
data_train = data_train.drop(['Cabin']      , axis=1)
data_train = data_train.drop(['PassengerId'], axis=1)
data_train = data_train.drop(['Ticket']     , axis=1)
data_train = data_train.drop(['Name']       , axis=1)

data_test  =  data_test.drop(['Cabin']      , axis=1)
data_test  =  data_test.drop(['PassengerId'], axis=1)
data_test  =  data_test.drop(['Ticket']     , axis=1)
data_test  =  data_test.drop(['Name']       , axis=1)


data_train.info()
combined_data = [data_train,data_test]


# Further clean the data
most_common_port = data_train.Embarked.dropna().mode()[0]
average_age      = data_train.Age.dropna().mean()
average_fare     = data_train.Fare.dropna().mean()
for dataset in combined_data:
    dataset['Age']      = dataset['Age'].fillna(average_age)
    dataset['Fare'] = dataset['Fare'].fillna(average_fare)
    dataset['Sex']      = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].fillna(most_common_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


X_train = data_train.drop("Survived", axis=1)
Y_train = data_train["Survived"]
X_test  = data_test

# Loop through classifiers and perform k-fold cross validation
classifiers = [svm.SVC(kernel='linear', C=1)]
class_names = ['SVM']

classifier_scores = []
print("Classifier k-fold score")
for i,clf in enumerate(classifiers):
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    print("%s %.2f" % (class_names[i],scores.mean()*100))
    classifier_scores.append(scores.mean()*100)

    # Train with all the data?
    clf.fit(X_train,Y_train)

    # Predict the Results with actual test data and generate CSV that
    # Kaggle needs to actually score
    results = clf.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": data_test_orig["PassengerId"],
        "Survived": results
    })

if GEN_OUTPUT:
    submission.to_csv('./output/submission.csv', index=False)

'''