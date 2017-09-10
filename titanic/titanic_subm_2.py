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


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, target='Survived'):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    #print(pd.Series(alg.booster().get_fscore()))
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
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
    if 'Sex' in tdata.columns:
        tdata = tdata.drop('Sex', axis=1)
    if 'Embarked' in tdata.columns:
        tdata = tdata.drop('Embarked', axis=1)
    for x in ['Name', 'Ticket', 'Cabin','EmbarkedC', 'SibSp', 'Parch', 'Fare']:
        if x in tdata.columns:
            tdata = tdata.drop(x, axis=1)

    tdata.to_csv('../data/titanic/out_'+os.path.basename(file))
    return tdata

def print_score(clf, X, y):

    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)

def print_sol(clf, test_df, output_file, passenger_ids):
    if 'PassengerId' in test_df.columns:
        test_df = test_df.drop('PassengerId', axis=1)
    yp = clf.predict(test_df)
    dsol = pd.DataFrame()
    dsol['PassengerId'] = passenger_ids
    dsol['Survived'] = yp

    dsol = dsol.set_index('PassengerId')
    dsol.to_csv(output_file, index_label='PassengerId')

train_df =  get_df('../data/titanic/train.csv')
test_df  =  get_df('../data/titanic/test.csv')

gen_fname = str(datetime.datetime.now()).replace(' ', '_') + '.log'
f = open('../logs/' + gen_fname, 'w')
IDcol = 'PassengerId'
target = 'Survived'
passengerIds = test_df[IDcol]


predictors = [x for x in train_df.columns if x not in [target, IDcol]]
"""
xgb1 = XGBClassifier(
 learning_rate =0.05,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.9,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 reg_alpha = 0.001,
 seed=27)
modelfit(xgb1, train_df, predictors)


print_sol(xgb1, test_df, '../data/titanic/xgb6.csv', passengerIds)

param_test1 = {
# 'max_depth':list(range(3,10,2)),
# 'min_child_weight':list(range(1,6,2))

  #  'gamma': [i / 10.0 for i in range(0, 5)]

#'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[i/10.0 for i in range(6,10)]
#'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
   'learning_rate' : [0.05,0.1,0.15,0.2,0.25,0.3]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140,
                                                   max_depth=9,
                                                   min_child_weight=5,
  subsample=0.9, colsample_bytree=0.8, gamma=0,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27,  reg_alpha = 0.001),
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_df[predictors],train_df[target])
print(gsearch1.best_params_, gsearch1.best_score_)

print_sol(gsearch1, test_df, '../data/titanic/xgb5.csv', passengerIds)

param_test2 = {
    'n_neighbors' : list(range(1,50))
}
ksearch1 = GridSearchCV(estimator = KNeighborsClassifier(),
                        param_grid=param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
ksearch1.fit(train_df[predictors],train_df[target])
print(ksearch1.best_params_, ksearch1.best_score_)

ksearch2 =  KNeighborsClassifier(10)
ksearch2.fit(train_df[predictors],train_df[target])
print_sol(ksearch2, test_df, '../data/titanic/knn3.csv', passengerIds)

"""

"""
param_test3 = {
    'n_estimators' : list(range(35,50)),
    'max_features' : list(range(2,7)),
    'min_samples_split' : list(range(2,6))
}


rsearch1 = GridSearchCV(estimator = RandomForestClassifier(criterion='entropy'),
                        param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
rsearch1.fit(train_df[predictors],train_df[target])
print(rsearch1 .best_params_, rsearch1 .best_score_)
"""
rsearch2 =  RandomForestClassifier(n_estimators=41, max_features=2,criterion='entropy', min_samples_split= 4)
rsearch2.fit(train_df[predictors],train_df[target])
print_sol(rsearch2, test_df, '../data/titanic/rand_2.csv', passengerIds)
print_score(rsearch2,train_df[predictors],train_df[target] )

rsearch3 =  RandomForestClassifier(n_estimators=41, max_features=3,criterion='entropy', min_samples_split= 4)
rsearch3.fit(train_df[predictors],train_df[target])
print_sol(rsearch3, test_df, '../data/titanic/rand_3.csv', passengerIds)
print_score(rsearch3,train_df[predictors],train_df[target] )
