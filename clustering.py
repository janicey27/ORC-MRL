# -*- coding: utf-8 -*-
"""
This file contains the functions to generate and perform the MDP clustering

algorithm on data for the MIT-Lahey Opioids project.

Created on Sun Mar  1 18:48:20 2020

@author: omars
"""

#################################################################
# Load Libraries
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from collections import Counter
from itertools import groupby
from operator import itemgetter
from testing import *
#################################################################


#################################################################
# Funtions for Initialization

# defaultNormal() takes as argument n, the number of laws to generate, p the
# dimension of the features, and sigma the correlation matrix
# Return a list of size n of Normal Random Variables of p dimensions
def defaultNormal(n,  # integer: number of observations
                  p=2,  # integer: number of dimensions
                  sigma=[[0.1, 0], [0, 0.1]]):  # array: correlation matrix
    def sample(mu):
        return(lambda: np.random.multivariate_normal(
            p*[mu], sigma))
    return [sample(mu) for mu in range(n)]

def UnifNormal(n,  # integer: number of observations
                  p=2,  # integer: number of dimensions
                  sigma=[[0.1, 0], [0, 0.1]]):  # array: correlation matrix
    def sample(mu):
        return(lambda: np.random.multivariate_normal(
            mu, sigma))
    return np.array([sample([mu1,mu2]) for mu1 in range(int(np.sqrt(n)+1)+1) for mu2 in range(int(np.sqrt(n))+1)])[:n]

# transformSamples() converts samples generated by MDPTools into a dataframe
def transformSamples(samples,  # array: samples generated bu MDPTools
                     pfeatures):  # integer: number of features
    n = len(samples)
    df1 = pd.DataFrame([samples[i][:2] for i in range(n)], columns=[
            'ID', 'TIME'])
    features = np.array([samples[i][2] for i in range(n)])
    df2 = pd.DataFrame(features, columns=['FEATURE_' + str(
            i+1) for i in range(pfeatures)])
    df3 = pd.DataFrame([samples[i][-3:] for i in range(n)], columns=[
            'ACTION', 'RISK', 'OG_CLUSTER'])
    df = pd.concat([df1, df2, df3], sort=False, axis=1)
    return(df)


# initializeClusters() takes as input a dataframe, a time horizon T,
# a clustering algorithm, a number of clusters n_clusters,
# and a random seed (optional) and returns a dataframe
# with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
def initializeClusters(df,  # pandas dataFrame: MUST contain a "RISK" column
                       clustering='KMeans',  # string: clustering algorithm
                       n_clusters=8,  # number of clusters
                       random_state=0):  # random seed for the clustering
    df = df.copy()
    if clustering == 'KMeans':
        output = KMeans(
                n_clusters=n_clusters, random_state=random_state).fit(
                        np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Agglomerative':
        output = AgglomerativeClustering(
            n_clusters=n_clusters).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Birch':
        output = Birch(
            n_clusters=n_clusters).fit(
                    np.array(df.RISK).reshape(-1, 1)).labels_
    else:
        output = LabelEncoder().fit_transform(np.array(df.RISK).reshape(-1, 1))
    df['CLUSTER'] = output
    df['NEXT_CLUSTER'] = df['CLUSTER'].shift(-1)
    df.loc[df['ID'] != df['ID'].shift(-1), 'NEXT_CLUSTER'] = 'None'
    return(df)
#################################################################


#################################################################
# Function for the Iterations

# findConstradiction() takes as input a dataframe and returns the tuple with
# initial cluster and action that have the most number of contradictions or
# (-1, -1) if no such cluster existss
def findContradiction(df, # pandas dataFrame
                      th): # integer: threshold split size
    X = df.loc[:, ['CLUSTER', 'NEXT_CLUSTER', 'ACTION']]
    X = X[X.NEXT_CLUSTER != 'None']
    count = X.groupby(['CLUSTER', 'ACTION'])['NEXT_CLUSTER'].nunique()
    contradictions = list(count[list(count > 1)].index)
    
    if len(contradictions) > 0:
        ncontradictions = [sum(list(X.query('CLUSTER == @i[0]').query(
                'ACTION == @i[1]').groupby('NEXT_CLUSTER')['ACTION'].count().
            sort_values(ascending=False).ravel())[1:]) for i in contradictions]
        if max(ncontradictions) > th:
            selectedCont = contradictions[ncontradictions.index(
                    max(ncontradictions))]
            return(selectedCont)
    
    return((-1, -1))


# contradiction() outputs one found contradiction given a dataframe,
# a cluster and a an action or (None, None) if none is found
def contradiction(df,  # pandas dataFrame
                  i,  # integer: initial clusters
                  a):  # integer: action taken
    nc = list(df.query('CLUSTER == @i').query(
            'ACTION == @a').query('NEXT_CLUSTER != "None"')['NEXT_CLUSTER'])
    if len(nc) == 1:
        return (None, None)
    else:
        return a, multimode(nc)[0]

# multimode() returns a list of the most frequently occurring values. 
# Will return more than one result if there are multiple modes 
# or an empty list if *data* is empty.
def multimode(data):
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))

# split() takes as input a dataframe, an initial cluster, an action, a target
# cluster that is a contradiction c, a time horizon T, then number of features,
# and an iterator k (that is the indexer of the next cluster), as well as the
# predictive classification algorithm used
# and returns a new dataframe with the contradiction resolved
def split(df,  # pandas dataFrame
          i,  # integer: initial cluster
          a,  # integer: action taken
          c,  # integer: target cluster
          pfeatures,  # integer: number of features
          k,  # integer: intedexer for next cluster
          classification='LogisticRegression'):  # string: classification aglo

    g1 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] == c)]
    g2 = df[(df['CLUSTER'] == i) & (
            df['ACTION'] == a) & (df['NEXT_CLUSTER'] != c) & (
                    df['NEXT_CLUSTER'] != 'None')]
    g3 = df[(df['CLUSTER'] == i) & (
            ((df['ACTION'] == a) & (df['NEXT_CLUSTER'] == 'None')) | (
                    df['ACTION'] != a))]
    groups = [g1, g2, g3]
    data = {}

    for j in range(len(groups)):

        d = pd.DataFrame(groups[j].iloc[:, 2:2+pfeatures].values.tolist())

        data[j] = d

    data[0].insert(data[0].shape[1], "GROUP", np.zeros(data[0].shape[0]))
    data[1].insert(data[1].shape[1], "GROUP", np.ones(data[1].shape[0]))

    training = pd.concat([data[0], data[1]])

    tr_X = training.iloc[:, :-1]
    tr_y = training.iloc[:, -1:]

    if classification == 'LogisticRegression':
        m = LogisticRegression(solver='liblinear')
    elif classification == 'LogisticRegressionCV':
        m = LogisticRegressionCV()
    elif classification == 'DecisionTreeClassifier':
        m = DecisionTreeClassifier()
    elif classification == 'RandomForestClassifier':
        m = RandomForestClassifier()
    #elif classification == 'XGBClassifier':
        #m = XGBClassifier()        
    else:
        m = LogisticRegression(solver='liblinear')
    
    
    m.fit(tr_X, tr_y.values.ravel())


    ids = g2.index.values

    test_X = data[2]

    if len(test_X) != 0:
        Y = m.predict(test_X)
        g3.insert(g3.shape[1], "GROUP", Y.ravel())
        id2 = g3.loc[g3["GROUP"] == 1].index.values
        ids = np.concatenate((ids, id2))
    

    '''
    df.loc[df.index.isin(ids), 'CLUSTER'] = k
    newids = ids-1
    newids = np.where((newids%T) != (T-1), newids, -1)
    df.loc[df.index.isin(newids), 'NEXT_CLUSTER'] = k
    '''
    
    df.loc[df.index.isin(ids), 'CLUSTER'] = k
    newids = ids-1
    df.loc[(df.index.isin(newids)) & 
           (df['ID']== df['ID'].shift(-1)), 'NEXT_CLUSTER'] = k

    return(df)


# splitter() is the wrap-up function. Takes as parameters a dataframe df,
# a time-horizon T, a number of features pfeatures, an indexer k, and a max
# number of iterations and performs the algorithm until all contradictions are
# resolved or until the max number of iterations is reached
# Returns the final resulting dataframe

def splitter(df,  # pandas dataFrame
             pfeatures,  # integer: number of features
             k,  # integer: indexer
             th, # integer: threshold for minimum split
             df_test,
             classification='LogisticRegression',  # string: classification alg
             it=6,
             OutputFlag = 1):  # integer: max number of iterations

    nc = k
    df_new = deepcopy(df)
    for i in range(it):
        cont = False
        c, a = findContradiction(df_new, th)
        if c != -1:
            if OutputFlag == 1:
                print(df_new.groupby(
                            ['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())
            
            a, b = contradiction(df_new, c, a)
            
            if OutputFlag == 1:
                print(c, a, b)
            df_new = split(df_new, c, a, b, pfeatures, nc, classification)
            model = predict_cluster(df_new, pfeatures)
            print('training value error:', training_value_error(df_new))
            print('testing value error:', testing_value_error(df_test, df_new, model, pfeatures))
            print('training accuracy:', training_accuracy(df_new))
            print('predictions:', get_predictions(df_new))
            #print(df_new.head())
            cont = True
            nc += 1
        if not cont:
            break
    if OutputFlag == 1:
        print(df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())

    return(df_new)

#################################################################

