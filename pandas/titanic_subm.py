import pandas as pd

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

def get_clf_fit(train_df):
    def get_X_y(data):
        X = data.drop('Survived', axis=1)
        y = data['Survived']
        return (X, y)

    X, y = get_X_y(train_df)

    from sklearn import svm
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X, y)
    return clf

def print_sol(clf, test_df, output_file):
    yp = clf.predict(test_df)
    dsol = pd.DataFrame()
    dsol['PassengerId'] = test_df['PassengerId']
    dsol['Survived'] = yp

    dsol = dsol.set_index('PassengerId')
    dsol.to_csv(output_file, index_label='PassengerId')

train_df =  get_df('../data/titanic/train.csv')
test_df  =  get_df('../data/titanic/test.csv')
clf      =  get_clf_fit(train_df )
print_sol(clf, test_df, '../data/titanic/svm_submiss_1.csv')



