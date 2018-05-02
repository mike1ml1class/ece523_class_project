#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import necessary libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
'''
 Clean, fill, fix and create new features from the data:

     Age, Cabin, and Embarked have missing data
     Cabin has too much missing data to be useful, so that will be dropped
     PassengerID, Name, and Ticket will also be dropped
     Take average of age and replace missing age values with average
     Embared will be changed to most common occurance
     Change male and female strings to 0 or 1
     Changed point of depature strings to 0, 1, or 2.

     This will leave us with numeric data to use in ML models
     Test data has a missing value for Fare, this will be replaced
     by the average fare
'''
def fix_fill_convert(x_tr, x_te,ALT_DATA):

    # Put data in a list
    data = [x_tr,x_te]

    average_fare     = x_tr.Fare.dropna().mean()
    # Loop over each data frame
    for idx,df in enumerate(data):

        # Remove - Dont seem to be correlated to survival
        df  =  df.drop(['Cabin']      , axis=1)
        df  =  df.drop(['PassengerId'], axis=1)
        df  =  df.drop(['Ticket']     , axis=1)

        # Convert categorical to numerical
        df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

        # fill mising value of fare with average fare
        df['Fare'] = df['Fare'].fillna(average_fare)

        if ALT_DATA:
            # Extract each person's title from their name
            df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

            # Replace uncommon titles with "Rare"
            df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
               'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

            # Replace old-world names with modern ones
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')

            # Convert categorical to numerical
            title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
            df['Title'] = df['Title'].fillna(0)
            df['Title'] = df['Title'].map(title_mapping)

        # Remove Name, don't need it anymore
        df = df.drop(['Name'], axis=1)

        # Get unique values
        u_sex = len(np.unique(df['Sex']))
        u_pclass = len(np.unique(df['Pclass']))

        # Initialize
        guess_ages = np.zeros((u_sex,u_pclass))

        # Loop over the unique values to estimate age in each sex,pclass bin
        for iS in range(0,u_sex):
            for iP in range(0,u_pclass):

                guess_df = df[(df['Sex'] == iS) & (df['Pclass'] == iP+1)]['Age'].dropna()

                age_guess = guess_df.median()

                # Convert to nearest .5 age
                guess_ages[iS,iP] = int( age_guess/0.5 + 0.5 ) * 0.5

                # Replace missing ages with guessed age
                df.loc[ (df.Age.isnull()) \
                      & (df.Sex == iS) \
                      & (df.Pclass == iP+1),'Age'] = guess_ages[iS,iP]

        # Convert type
        df['Age'] = df['Age'].astype(int)

        # Bin continious age into "buckets" - TODO: Determine best number bins
        #df['AgeBin'] = pd.cut(df['Age'], 8)

        #df[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='AgeBin', ascending=True)

        #          AgeBin  Survived
        #0  (-0.08, 16.0]  0.550000
        #1   (16.0, 32.0]  0.337374
        #2   (32.0, 48.0]  0.412037
        #3   (48.0, 64.0]  0.434783
        #4   (64.0, 80.0]  0.090909

        #          AgeBin  Survived
        #0  (-0.08, 10.0]  0.593750
        #1   (10.0, 20.0]  0.379310
        #2   (20.0, 30.0]  0.322751
        #3   (30.0, 40.0]  0.448649
        #4   (40.0, 50.0]  0.392857
        #5   (50.0, 60.0]  0.404762
        #6   (60.0, 70.0]  0.222222
        #7   (70.0, 80.0]  0.250000


        # Convert the category bins into numerical values - TODO: figure out how to automate this
        #df.loc[ df['Age'] <= 16                     , 'Age'] = 0
        #df.loc[(df['Age'] >  16) & (df['Age'] <= 32), 'Age'] = 1
        #df.loc[(df['Age'] >  32) & (df['Age'] <= 48), 'Age'] = 2
        #df.loc[(df['Age'] >  48) & (df['Age'] <= 64), 'Age'] = 3
        #df.loc[ df['Age'] >  64                     , 'Age'] = 4

        # Convert the category bins into numerical values - TODO: figure out how to automate this
        #df.loc[ df['Age'] <= 10  , 'Age'] = 0
        #df.loc[(df['Age'] >  10) , 'Age'] = 1

        if ALT_DATA:
            df.loc[ df['Age'] <= 10                     , 'Age'] = 0
            df.loc[(df['Age'] >  10) & (df['Age'] <= 20), 'Age'] = 1
            df.loc[(df['Age'] >  20) & (df['Age'] <= 30), 'Age'] = 2
            df.loc[(df['Age'] >  30) & (df['Age'] <= 40), 'Age'] = 3
            df.loc[(df['Age'] >  40) & (df['Age'] <= 50), 'Age'] = 4
            df.loc[(df['Age'] >  50) & (df['Age'] <= 60), 'Age'] = 5
            df.loc[(df['Age'] >  60) & (df['Age'] <= 70), 'Age'] = 6
            df.loc[ df['Age'] >  70                     , 'Age'] = 7


        if ALT_DATA:
            # Create FamilySize by combining SibSp (num siblings/spouses)
            # and Parch (num parents/children)
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

            # Remove SibSp and Parch
            df = df.drop(['SibSp'] , axis=1)
            df = df.drop(['Parch'] , axis=1)

            # Create Solo for those traveling alone
            df['Solo'] = 0
            df.loc[df['FamilySize'] == 1, 'Solo'] = 1

            # Remove FamilySize too
            #df = df.drop(['FamilySize'])

        # Create a new feature by combining
        #if ALT_DATA:
            #df['Age*Class'] = df.Age * df.Pclass

        # Fill missing port of departure with the most common and convert to numerical
        port_mode = df.Embarked.dropna().mode()[0]
        df['Embarked'] = df['Embarked'].fillna(port_mode)
        df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        # Bin continious fare into "buckets" - TODO: Determine best number bins
        #df['FareBins'] = pd.qcut(df['Fare'], 10)
        #df[['FareBins', 'Survived']].groupby(['FareBins'], as_index=False).mean().sort_values(by='FareBins', ascending=True)

        #          FareBins  Survived
        #0   (-0.001, 7.91]  0.197309
        #1   (7.91, 14.454]  0.303571
        #2   (14.454, 31.0]  0.454955
        #3  (31.0, 512.329]  0.581081

        #            FareBins  Survived
        #0     (-0.001, 7.55]  0.141304
        #1      (7.55, 7.854]  0.298851
        #2      (7.854, 8.05]  0.179245
        #3       (8.05, 10.5]  0.230769
        #4     (10.5, 14.454]  0.428571
        #5   (14.454, 21.679]  0.420455
        #6     (21.679, 27.0]  0.516854
        #7     (27.0, 39.688]  0.373626
        #8   (39.688, 77.958]  0.528090
        #9  (77.958, 512.329]  0.758621

        if ALT_DATA:
            # Convert the category bins into numerical values - TODO: figure out how to automate this
            df.loc[ df['Fare'] <= 7.55                            , 'Fare'] = 0
            df.loc[(df['Fare'] >  7.55  ) & (df['Fare'] <= 7.845 ), 'Fare'] = 1
            df.loc[(df['Fare'] >  7.854 ) & (df['Fare'] <= 8.05  ), 'Fare'] = 2
            df.loc[(df['Fare'] >  8.05  ) & (df['Fare'] <= 10.5  ), 'Fare'] = 3
            df.loc[(df['Fare'] >  10.5  ) & (df['Fare'] <= 14.454), 'Fare'] = 4
            df.loc[(df['Fare'] >  14.454) & (df['Fare'] <= 21.679), 'Fare'] = 5
            df.loc[(df['Fare'] >  21.679) & (df['Fare'] <= 27.0  ), 'Fare'] = 6
            df.loc[(df['Fare'] >  27.0  ) & (df['Fare'] <= 39.688), 'Fare'] = 7
            df.loc[(df['Fare'] >  39.688) & (df['Fare'] <= 77.958), 'Fare'] = 8
            df.loc[ df['Fare'] >  77.958                          , 'Fare'] = 9

        # Overwrite the original df
        data[idx] = df


    return data


def visualize_data(x_tr,x_te):

    print('\nTraining Data:')
    print( x_tr.info() )

    print('\nTesting Data:')
    print( x_te.info() )

    print('\nDescribe Training:')
    print( x_tr.describe() )



    # Do a bit of analysis by isolating features
    print('\n')
    out = x_tr[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(out)

    print('\n')
    out = x_tr[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(out)

    #print('\n')
    #out = x_tr[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    #print(out)

    print('\n')
    out = x_tr[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(out)

    print('\n')
    out = x_tr[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(out)

    print('\n')
    x_tr['FamilySize'] = x_tr['SibSp'] + x_tr['Parch'] + 1
    out = x_tr[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(out)


    f,ax = plt.subplots(1,2,figsize=(8,4),sharey=True)
    #ax = x_te.hist(column='Age',by='Survived',sharey=True,xrot=45,bins=20,ec='k')
    x_tr.hist(ax=ax,column='Age',by='Survived',rot=0,bins=20,ec='k')
    ax[0].set_xlabel('Age')
    ax[1].set_xlabel('Age')
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_title('Survived = 0')
    ax[1].set_title('Survived = 1')
    plt.tight_layout()
    f.savefig('Age_hist')

    f,ax = plt.subplots(3,2,figsize=(8,8),sharex='col',sharey='row')
    #x_te.hist(ax=ax,column='Age',by=['Pclass','Survived'],sharey=True,sharex=True,xrot=45,bins=20,ec='k')
    x_tr.hist(ax=ax,column='Age',by=['Pclass','Survived'],rot=0,bins=20,ec='k')
    ax[0,0].set_title('Pclass = 1,Survived = 0')
    ax[0,1].set_title('Pclass = 1,Survived = 1')
    ax[1,0].set_title('Pclass = 2,Survived = 0')
    ax[1,1].set_title('Pclass = 2,Survived = 1')
    ax[2,0].set_title('Pclass = 3,Survived = 0')
    ax[2,1].set_title('Pclass = 3,Survived = 1')
    ax[2,0].set_xlabel('Age')
    ax[2,1].set_xlabel('Age')
    ax[0,0].grid(True)
    ax[0,1].grid(True)
    ax[1,0].grid(True)
    ax[1,1].grid(True)
    ax[2,0].grid(True)
    ax[2,1].grid(True)
    plt.tight_layout()
    f.savefig('Pclass_hist')



    # clean the data for the sake of the plot
    #copy data to temp_tr so actual data set does not change and mess with the rest of the script
    port_mode = x_tr.Embarked.dropna().mode()[0]
    temp_tr = x_tr.copy()
    temp_tr['Embarked'] = temp_tr['Embarked'].fillna(port_mode)
    #temp_tr['Embarked'] = temp_tr['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    #temp_tr['Sex'] = temp_tr['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#    f,ax = plt.subplots(3,2,figsize=(8,8),sharey='row')
#     #x_te.hist(ax=ax,column='Age',by=['Pclass','Survived'],sharey=True,sharex=True,xrot=45,bins=20,ec='k')
#    temp_tr.hist(ax=ax,column='Fare',by=['Embarked','Survived'],rot=0,bins=20,ec='k')
#    ax[0,0].set_title('Embarked = S,Survived = 0')
#    ax[0,1].set_title('Embarked = S,Survived = 1')
#    ax[1,0].set_title('Embarked = C,Survived = 0')
#    ax[1,1].set_title('Embarked = C,Survived = 1')
#    ax[2,0].set_title('Embarked = Q,Survived = 0')
#    ax[2,1].set_title('Embarked = Q,Survived = 1')
#    ax[2,0].set_xlabel('Fare')
#    ax[2,1].set_xlabel('Fare')
#    ax[0,0].grid(True)
#    ax[0,1].grid(True)
#    ax[1,0].grid(True)
#    ax[1,1].grid(True)
#    ax[2,0].grid(True)
#    ax[2,1].grid(True)
#    plt.tight_layout()
#    f.savefig('Fare_hist')

    grid = sns.FacetGrid(temp_tr, row='Embarked', col='Survived',size=2.5,aspect=1.5)
    g = grid.map(sns.barplot, 'Sex','Fare',ci=None)
    sns.set_style("whitegrid", {'axes.grid' : True,"axes.edgecolor": "black"})
    sns.set_context("paper", rc={'lines.edgecolor':'black'})
    g.savefig('Fare_hist')

def normalize_column(data,field):
    scaler = MinMaxScaler()
    data[field] = scaler.fit_transform(data[field].values.reshape(-1,1))
    return data

def create_submission(passenger_id,survived,filename):
    submission = pd.DataFrame({"PassengerId": passenger_id,
                               "Survived"   : survived      })
    submission.to_csv(filename, index=False)