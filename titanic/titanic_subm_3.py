import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import xgboost
import os
import sys
import datetime
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
best_score, best_clf = 0, None

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return 0



    #plt.ylabel('Feature Importance Score')

def get_df(file):
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    def convSex(row):
        return 1 if row['Sex'] == 'male' else 0

    def convEmbarked(row):
        return {'S': 0, 'C': 1, 'Q': 2}[row['Embarked']]


    # replacing all titles with mr, mrs, miss, master
    def conv_deck(x):
        deck = x['Deck']
        if deck in cabin_list:
            return cabin_list.index(deck)
        else:
            return 0

    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 0
        elif title in ['Countess', 'Mme']:
            return 1
        elif title in ['Mlle', 'Miss']:
            return 2
        elif title == 'Dr':
            if x['Sex'] == 'male':
                return 3
            else:
                return 4
        else:
            return 5

    tdata = pd.read_csv(file)
    tdata['Cabin'] =     tdata['Cabin'].fillna(0)

    tdata['Deck'] = tdata ['Cabin'].map(lambda x: substrings_in_string(x, cabin_list) if x != 0 else 0)
    tdata['Deck'] = tdata.apply(conv_deck, axis=1)

    med_age = tdata.median()['Age']

    tdata['Age'] = tdata['Age'].fillna(med_age)

    tdata['Title'] = tdata['Name'].map(lambda x: substrings_in_string(x, title_list))

    tdata['Title'] = tdata.apply(replace_titles, axis=1)

    med_fare = tdata.median()['Fare']

    tdata['Fare'] = tdata['Fare'].fillna(med_fare)
    tdata['Embarked'] = tdata['Embarked'].fillna('C')



    tdata['SexC'] = tdata.apply(convSex, axis=1)

    tdata['Family_Size'] = tdata['SibSp'] + tdata['Parch']
    tdata['Fare_Per_Person'] = tdata['Fare'] / (tdata['Family_Size'] + 1)

    tdata['EmbarkedC'] = tdata.apply(convEmbarked, axis=1)


    tdata.to_csv('data/out_'+os.path.basename(file))
    return tdata

def step_df(tdata):
    dummy_sex = pd.get_dummies(tdata['SexC'], prefix='SexC')
    dummy_embarked = pd.get_dummies(tdata['EmbarkedC'], prefix='EmbarkedC')
    dummy_title = pd.get_dummies(tdata['Title'], prefix='Title')
    dummy_pclass = pd.get_dummies(tdata['Pclass'], prefix='Pclass')
    dummy_pclass = pd.get_dummies(tdata['Deck'], prefix='Deck')

    tdata = pd.concat([tdata, dummy_sex, dummy_embarked, dummy_title, dummy_pclass], axis=1)


    if 'PassengerId' in tdata.columns:
        tdata = tdata.drop('PassengerId', axis=1)

    return tdata

def print_score(clf, X, y):

    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)

def print_sol(clf, test_df, output_file, passenger_ids):

    yp = clf.predict(test_df)
    dsol = pd.DataFrame()
    dsol['PassengerId'] = passenger_ids
    dsol['Survived'] = yp

    dsol = dsol.set_index('PassengerId')
    dsol.to_csv(output_file, index_label='PassengerId')

def classifier():
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    import numpy as np
    return SVC(kernel='linear', C=1000, gamma=1000)


train_df =  get_df('data/train.csv')
test_df  =  get_df('data/test.csv')

passenger_ids = test_df['PassengerId']
survived = train_df['Survived']

train_df =  step_df(train_df)
test_df  =  step_df(test_df)


rf = classifier()


print(train_df.head())
print(train_df.columns)
import numpy as np


rel_columns = [ 'Age', 'SibSp', 'Parch',
       'Fare',  'SexC_0', 'SexC_1', 'EmbarkedC_0',
       'EmbarkedC_1', 'EmbarkedC_2', 'Title_0', 'Title_1', 'Title_2',
       'Title_3', 'Title_4', 'Title_5', 'Deck_0', 'Deck_1', 'Deck_2', 'Deck_3',
       'Deck_4', 'Deck_5', 'Deck_6', 'Deck_7']
X_train = np.array(train_df[rel_columns])
y_train = np.array(survived)

misscolumns = [x for x in rel_columns if x not in test_df.columns]
print(misscolumns)

for ms in misscolumns:
    print(ms)
    test_df[ms] = 0

X_test = np.array(test_df[rel_columns])
rf.fit(X_train,y_train)

print_score(rf, X_train, y_train)
print_sol(rf, X_test, 'data/gsvc1.csv', passenger_ids)