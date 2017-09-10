import json
import pandas as pd
import math
import numpy as np


def load_submissions():

    with open('old/submissions.json', 'r') as f:
        subms_json = json.load(f)
    subms  = []
    reslts = []
    for k,v in subms_json.items():
        df = pd.read_csv('old/'+k)
        subms.append(df)
        reslts.append(v)
    return subms, reslts
#    for subm in subms:
#        print(subm.groupby('Survived').count())
#    for restl in reslts:
#        print(restl)


def comparesubm(ys, subms, reslts):
    sumtot = 0
    for i in range(len(subms)):
        subm = subms[i]
        reslt = reslts[i]
        dars = (sum(abs(ys-subm.iloc[:,1])))
        invres = 418-reslt*418
        #print('Drs {} : Invrs {}'.format(dars,invres))
        sumtot += abs(dars-invres)
    return sumtot


def showrow():
    subms, results = load_submissions()
    df = pd.DataFrame()
    for subm in subms:
        df = pd.concat([df,subm.iloc[:,1]],axis=1)

    df.to_csv('all_subm.csv')

def get_probs():
    subms, results = load_submissions()
    pgrid = subms[0].iloc[:,0]
    df = pd.DataFrame()
    df = pd.concat([df,pgrid,pd.DataFrame(np.zeros(pgrid.shape))],axis=1)
    #print(df)
    for i in range(len(subms)):
        subm = subms[i]
        reslt = results[i]
        df.iloc[:,1] = df.iloc[:,1] + subm.iloc[:,1]*reslt
    df.iloc[:, 1] = df.iloc[:, 1] / len(subms)

    return df

#get_probs()
#showrow()