import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import xgboost
import os
import sys
import datetime
from xgboost import XGBClassifier

best_score, best_clf = 0, None

def get_df(file):
    tdata = pd.read_csv(file)
    if ('Cabin' in tdata.columns):
        tdata = tdata.drop('Cabin', axis=1)

    med_age = tdata.median()['Age']

    tdata['Age'] = tdata['Age'].fillna(med_age)

    med_fare = tdata.median()['Fare']

    tdata['Fare'] = tdata['Fare'].fillna(med_fare)
    tdata['Embarked'] = tdata['Embarked'].fillna('C')
    for x in ['Name', 'Ticket', 'Cabin']:
        if x in tdata.columns:
            tdata = tdata.drop(x, axis=1)


    def convSex(row):
        return 1 if row['Sex'] == 'male' else 0


    tdata['SexC'] = tdata.apply(convSex, axis=1)


    def convEmbarked(row):
        return {'S': 0, 'C': 1, 'Q': 2}[row['Embarked']]


    tdata['EmbarkedC'] = tdata.apply(convEmbarked, axis=1)
    if 'Sex' in tdata.columns:
        tdata = tdata.drop('Sex', axis=1)
    if 'Embarked' in tdata.columns:
        tdata = tdata.drop('Embarked', axis=1)



    return tdata


def do_sol(clf, train_df, test_df, outputfile, preprocess = None, f = sys.stdout):
    global best_score, best_clf

    def get_clf_fit(clf, train_df):
        global best_score, best_clf

        def get_X_y(data):
            X = data.drop('Survived', axis=1)
            y = data['Survived']
            return (X, y)

        def print_score(clf, X, y):
            global best_score, best_clf

            scores = cross_val_score(clf, X, y, cv=5)
            if (sum(scores) > best_score):
                best_score = sum(scores)
                best_clf = clf

            print(scores, file=f)

        X, y = get_X_y(train_df)
        clf.fit(X, y)
        print_score(clf,X,y)
        return clf

    def print_sol(clf, test_df, output_file, passenger_ids):
        yp = clf.predict(test_df)
        dsol = pd.DataFrame()
        dsol['PassengerId'] = passenger_ids
        dsol['Survived'] = yp

        dsol = dsol.set_index('PassengerId')
        dsol.to_csv(output_file, index_label='PassengerId')


    passenger_ids = test_df['PassengerId']

    if preprocess:
        train_df = preprocess(train_df)
        test_df  = preprocess(test_df)

    if not os.path.isfile(outputfile):

        print("Classifier initialized : {}".format(clf),file=f)
        clf = get_clf_fit(clf, train_df)
        print_sol(clf, test_df, outputfile, passenger_ids )
        print("Printed result to file : {}".format(clf, outputfile),file=f)
    else:

        print("File {} already exists - Skipping".format(outputfile),file=f)

def normalize_fields(data):
    data = data.drop(['EmbarkedC', 'SibSp', 'Parch', 'Fare', 'PassengerId'], axis=1 )
    return data

def improve_fields(data):
    data = normalize_fields(data)

    def convAge(row):
        if row['Age'] < 9:
            return 0
        elif row['Age'] < 14:
            return 1
        elif row['Age'] < 25:
            return 2
        else:
            return 3

    data['AgeC'] = data.apply(convAge, axis=1)
    data = data.drop('Age', axis=1)
    return data

train_df =  get_df('../data/titanic/train.csv')
test_df  =  get_df('../data/titanic/test.csv')

gen_fname = str(datetime.datetime.now()).replace(' ', '_') + '.log'
f = open('../logs/' + gen_fname, 'w')

#do_sol(svm.SVC(kernel='linear', C=1), train_df, test_df, '../data/titanic/svm_submiss_1.csv')
#do_sol(RandomForestClassifier(n_estimators=50), train_df , test_df, '../data/titanic/random_forest_1.csv',f=f)
#do_sol(RandomForestClassifier(n_estimators=50), train_df , test_df, '../data/titanic/random_forest_2.csv', normalize_fields,f=f)
#for i in range(3,50):
#    do_sol(RandomForestClassifier(n_estimators=i), train_df, test_df, '../data/titanic/random_forest_'+str(i)+'.csv', improve_fields,f=f)
#    do_sol(KNeighborsClassifier(i), train_df , test_df, '../data/titanic/knn_'+str(i)+'.csv', improve_fields,f=f)
do_sol(XGBClassifier(), train_df, test_df, '../data/titanic/xgb_0.csv', improve_fields, f=f)

print('===== BEST CLASSIFIER =====')
print(best_clf)
print('===== BEST SCORE =====')
print(best_score)
