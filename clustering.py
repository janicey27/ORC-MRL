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
import matplotlib as plt
from tqdm import tqdm #progress bar
import binascii
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
# with format: ['ID', 'TIME', *Features*, 'ACTION', 'RISK', 'OG_CLUSTER']
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


# split_train_test_by_id() takes in a dataframe of all the data, 
# returns Testing and Training dataset dataframes with the ratio of testing
# data defined by float test_ratio
def split_train_test_by_id(data, # dataframe: all the data
                           test_ratio, # float: portion of data for testing
                           id_column): # str: name of identifying ID column
    
    def test_set_check(identifier, test_ratio):
        return binascii.crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


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
#        params = {
#        'max_depth': [3, 4, 6, 10,None]
#        }
#        m = GridSearchCV(m, params,cv = 5)
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
# Plots the trajectory of testing metrics during splitting process
# Returns the final resulting dataframe
def splitter(df,  # pandas dataFrame
             pfeatures,  # integer: number of features
             k,  # integer: indexer
             th, # integer: threshold for minimum split
             df_test,
             classification='LogisticRegression',  # string: classification alg
             it=6,
             OutputFlag = 1,
             n=-1):  # integer: max number of iterations
    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    training_acc = []
    testing_acc = []
    testing_error = []
    training_error = []
    nc = k
    df_new = deepcopy(df)
    
    # Setting progress bar--------------
    split_bar = tqdm(range(it))
    split_bar.set_description("Splitting...")
    # Setting progress bar--------------
    for i in split_bar:
        split_bar.set_description("Splitting... |#Clusters:%s" %(nc))
        cont = False
        c, a = findContradiction(df_new, th)
        print('Iteration',i+1, '| #Clusters=',nc+1, '------------------------')
        if c != -1:
            if OutputFlag == 1:
                print('Cluster Content')
                print(df_new.groupby(
                            ['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())
            
            # finding contradictions and splitting
            a, b = contradiction(df_new, c, a)
            
            if OutputFlag == 1:
                print('Cluster splitted', c,'| Action causing contradiction:', a, '| Cluster most elements went to:', b)
            df_new = split(df_new, c, a, b, pfeatures, nc, classification)
            
            # error and accuracy calculations
            model = predict_cluster(df_new, pfeatures)
            R2_train = R2_value_training(df_new)
            R2_test = R2_value_testing(df_test, df_new, model, pfeatures)
            train_acc = training_accuracy(df_new)[0]
            test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
            train_error = training_value_error(df_new)
            test_error = testing_value_error(df_test, df_new, model, pfeatures)
            training_R2.append(R2_train)
            testing_R2.append(R2_test)
            training_acc.append(train_acc)
            testing_acc.append(test_acc)
            testing_error.append(test_error)
            training_error.append(train_error)
            
            # printing error and accuracy values
            if OutputFlag == 1:
                print('training value R2:', R2_train)
                print('testing value R2:', R2_test)
                print('training accuracy:', train_acc)
                print('testing accuracy:', test_acc)
                print('training value error:', train_error)
                print('testing value error:', test_error)
            #print('predictions:', get_predictions(df_new))
            #print(df_new.head())
            cont = True
            nc += 1
        if not cont:
            break
    if OutputFlag == 1:
        print(df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())
    
    
    # plotting functions
    ## Plotting accuracy and value R2
    fig1, ax1 = plt.subplots()
    its = np.arange(k+1, nc+1)
    ax1.plot(its, training_R2, label= "Training R2")
    ax1.plot(its, testing_R2, label = "Testing R2")
    ax1.plot(its, training_acc, label = "Training Accuracy")
    ax1.plot(its, testing_acc, label = "Testing Accuracy")
    if n>0:
        ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
    ax1.set_ylim(0,1)
    ax1.set_xlabel('# of Clusters')
    ax1.set_ylabel('R2 or Accuracy %')
    ax1.set_title('R2 and Accuracy During Splitting')
    ax1.legend()
    ## Plotting value error E((v_est - v_true)^2)
    fig2, ax2 = plt.subplots()
    ax2.plot(its, training_error, label = "Training Error")
    ax2.plot(its, testing_error, label = "Testing Error")
    if n>0:
        ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
    ax2.set_ylim(0)
    ax2.set_xlabel('# of Clusters')
    ax2.set_ylabel('Value error')
    ax2.set_title('Value error by number of clusters')
    ax2.legend()
    plt.show()
    
    return(df_new,training_R2,testing_R2)

#################################################################



# Splitter algorithm with cross-validation
def fit_CV(df,
          pfeatures,
          k,
          th,
          clustering,
          classification,
          n_iter,
          n_clusters,
          random_state,
          OutputFlag = 0,
          n=-1,
          cv=5):
    
    list_training_R2 = []
    list_testing_R2 = []
    data_size = df['ID'].max()
    
    cv_bar = tqdm(range(cv))
    cv_bar.set_description("Cross-Validation...")
    for i in cv_bar:
        cv_bar.set_description("Cross-Validation... | Test set # %")
        
        df_test = df[(df['ID']<(i+1)*data_size//cv) & (df['ID']>=i*data_size//cv)] #WARNING last datapoint not included
        df_train = df[~((df['ID']<(i+1)*data_size//cv) & (df['ID']>=i*data_size//cv))]
        #################################################################
        # Initialize Clusters
        df_init = initializeClusters(df_train,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                random_state=random_state)
        #################################################################
        
        #################################################################
        # Run Iterative Learning Algorithm
        
        df_new,training_R2,testing_R2 = splitter(df_init,
                                          pfeatures,
                                          k,
                                          th,
                                          df_test,
                                          classification,
                                          n_iter,
                                          OutputFlag = 0,
                                          n=n)
        list_training_R2.append(np.array(training_R2))
        list_testing_R2.append(np.array(testing_R2))
        
    cv_training_R2 = np.mean(np.array(list_training_R2),axis=0)
    cv_testing_R2 = np.mean(np.array(list_testing_R2),axis=0)
    
    fig1, ax1 = plt.subplots()
    its = np.arange(k+1,k+1+len(cv_training_R2))
    ax1.plot(its, cv_training_R2, label= "CV Training R2")
    ax1.plot(its, cv_testing_R2, label = "CV Testing R2")
#    ax1.plot(its, training_acc, label = "Training Accuracy")
#    ax1.plot(its, testing_acc, label = "Testing Accuracy")
    if n>0:
        ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
    ax1.set_ylim(0,1)
    ax1.set_xlabel('# of Clusters')
    ax1.set_ylabel('Mean CV R2 or Accuracy %')
    ax1.set_title('Mean CV R2 and Accuracy During Splitting')
    ax1.legend()
    
    return (list_training_R2,list_testing_R2)