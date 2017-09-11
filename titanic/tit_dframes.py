from abc import ABCMeta

import pandas as pd


def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return 0



def enrich(tdata):
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']

    def convSex(row):
        return 1 if row['Sex'] == 'male' else 0

    def convEmbarked(row):
        return {'S': 0, 'C': 1, 'Q': 2}[row['Embarked']]


    def def_age(row):
        if pd.isnull(row['Age']):
            if row['Pclass'] == 1:
                return 5
            elif row['Pclass'] == 2:
                return 10
            else:
                return 15
        else:
            return row['Age']
    # replacing all titles with mr, mrs, miss, master
    def conv_deck(x):
        deck = x['Deck']
        if deck in cabin_list:
            return cabin_list.index(deck)
        else:
            return 0

    def convAge(row):
        if row['Age'] < 9:
            return 0
        elif row['Age'] < 14:
            return 1
        elif row['Age'] < 25:
            return 2
        elif row['Age'] < 40:
            return 3

        else:
            return 4

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

    tdata['Cabin'] =     tdata['Cabin'].fillna(0)

    tdata['Deck'] = tdata ['Cabin'].map(lambda x: substrings_in_string(x, cabin_list) if x != 0 else 0)
    tdata['Deck'] = tdata.apply(conv_deck, axis=1)

    med_age = tdata.median()['Age']

    tdata['Age'] = tdata.apply(def_age, axis=1)

    tdata['Title'] = tdata['Name'].map(lambda x: substrings_in_string(x, title_list))

    tdata['Title'] = tdata.apply(replace_titles, axis=1)

    med_fare = tdata.median()['Fare']

    tdata['Fare'] = tdata['Fare'].fillna(med_fare)
    tdata['Embarked'] = tdata['Embarked'].fillna('C')
    tdata['AgeC'] = tdata.apply(convAge, axis=1)


    tdata['SexC'] = tdata.apply(convSex, axis=1)

    tdata['Family_Size'] = tdata['SibSp'] + tdata['Parch']
    tdata['Fare_Per_Person'] = tdata['Fare'] / (tdata['Family_Size'] + 1)

    tdata['EmbarkedC'] = tdata.apply(convEmbarked, axis=1)
    if 'Sex' in tdata.columns:
        tdata = tdata.drop('Sex', axis=1)
    if 'Embarked' in tdata.columns:
        tdata = tdata.drop('Embarked', axis=1)
    return tdata

def get_sol(clf, test_df ):
    if 'PassengerId' in test_df.columns:
        test_df = test_df.drop('PassengerId', axis=1)
    yp = clf.predict(test_df)
    return yp

def print_sol(yp, output_file, passenger_ids):
    dsol = pd.DataFrame()
    dsol['PassengerId'] = passenger_ids
    dsol['Survived'] = yp

    dsol = dsol.set_index('PassengerId')
    dsol.to_csv(output_file, index_label='PassengerId')
    return yp

train_df =  pd.read_csv('data/train.csv')
test_df  =  pd.read_csv('data/test.csv')

etrain_df = enrich(train_df)
etest_df = enrich(test_df)


def do_group(etrain_df):
    #ndf = etrain_df.groupby(['SexC','Pclass','AgeC','Survived'])['PassengerId'].count().reset_index(name="count")
    ndf = etrain_df.groupby(['SexC', 'Pclass', 'AgeC', 'Survived'])['PassengerId'].count().reset_index(name="Count")

    #ndf.columns = ['SexC', 'Pclass', 'AgeC', 'Survived', 'Count']

    print(ndf)
    #print(ndf.pivot(index='SexC', columns=['Pclass','AgeC', 'Survived'], values='Count'))


    ndf2 = ndf.set_index(['SexC', 'Pclass', 'AgeC', 'Survived'])
    print(ndf2)
    print(ndf2.loc[pd.IndexSlice[:,:,:,1],:])
    print(ndf2.pivot_table(index='SexC', columns=['Pclass','AgeC', 'Survived'], values='Count'))
    print(ndf2.pivot_table(index=['SexC','Pclass','AgeC'], columns=['Survived'], values='Count'))

    print(ndf2.unstack(['Survived']))
    print(ndf2.unstack(['AgeC', 'Survived']))

# print(ndf2)


def do_stuff():
    mtdf = etrain_df[etrain_df['SexC']==1]
    ftdf = etrain_df[etrain_df['SexC']==0]

    ndm = mtdf .groupby(['Pclass','AgeC','Survived'])['PassengerId'].count().reset_index(name="count")
    ndf = ftdf.groupby(['Pclass','AgeC','Survived'])['PassengerId'].count().reset_index(name="count")

    print(ndm[ndm['Survived']==1])
    print(ndm[ndm['Survived']==0])
    print(ndf[ndf['Survived']==1])
    print(ndf[ndf['Survived']==0])

do_group(etrain_df)