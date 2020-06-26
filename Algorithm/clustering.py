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
import matplotlib.pyplot as plt
import random
from datetime import datetime
from tqdm import tqdm #progress bar
import binascii
from copy import deepcopy
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
#from xgboost import XGBClassifier
from collections import Counter
from itertools import groupby
from operator import itemgetter

from testing import R2_value_training, training_value_error, training_accuracy, \
    predict_cluster, R2_value_testing, testing_value_error, testing_accuracy
#################################################################


#################################################################
# Funtions for Initialization


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
                       clustering='Agglomerative',  # string: clustering algorithm
                       n_clusters= None,
                       distance_threshold= 0.3,# number of clusters
                       random_state=0):  # random seed for the clustering
    df = df.copy()
    if clustering == 'KMeans':
        output = KMeans(
                n_clusters=n_clusters, random_state=random_state).fit(
                        np.array(df.RISK).reshape(-1, 1)).labels_
    elif clustering == 'Agglomerative':
        output = AgglomerativeClustering(
            n_clusters=n_clusters, distance_threshold = distance_threshold).fit(
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
          classification='LogisticRegression', # string: classification aglo
          split_classifier_params={'random_state':0}): # dict: of classifier params

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
        m = LogisticRegression(**split_classifier_params)
    elif classification == 'LogisticRegressionCV':
        m = LogisticRegressionCV(**split_classifier_params)
    elif classification == 'DecisionTreeClassifier':
        m = DecisionTreeClassifier(**split_classifier_params)
#        params = {
#        'max_depth': [3, 4, 6, 10,None]
#        }
#        m = GridSearchCV(m, params,cv = 5)
    elif classification == 'RandomForestClassifier':
        m = RandomForestClassifier(**split_classifier_params)
    #elif classification == 'XGBClassifier':
        #m = XGBClassifier()        
    elif classification == 'MLPClassifier':
        m = MLPClassifier(**split_classifier_params)
    elif classification == 'AdaBoostClassifier':
        m = AdaBoostClassifier(**split_classifier_params)
    else:
        m = LogisticRegression(**split_classifier_params)
    
    
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
             th, # integer: threshold for minimum split
             df_test = None, # df_test provided for cross validation
             testing = False, # True if we are cross validating
             max_k = 6, # int: max number of clusters
             classification='LogisticRegression',  # string: classification alg
             split_classifier_params = {'random_state':0}, # dict: classification params
             h=5,
             OutputFlag = 1,
             n=-1,
             plot = False):  
    # initializing lists for error & accuracy data
    training_R2 = []
    testing_R2 = []
    training_acc = []
    testing_acc = []
    testing_error = []
    training_error = []
    
    # determine if the problem has OG cluster
    if 'OG_CLUSTER' in df.columns:
        grid = True
    else:
        grid = False
    
    k = df['CLUSTER'].nunique() #initial number of clusters 
    nc = k #number of clusters
    
    df_new = deepcopy(df)
    
    # Setting progress bar--------------
    split_bar = tqdm(range(max_k-k))
    split_bar.set_description("Splitting...")
    # Setting progress bar--------------
    for i in split_bar:
        split_bar.set_description("Splitting... |#Clusters:%s" %(nc))
        cont = False
        c, a = findContradiction(df_new, th)
        if c != -1:
            if OutputFlag == 1:
                print('Cluster Content')
                print(df_new.groupby(
                            ['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())
            
            # finding contradictions and splitting
            a, b = contradiction(df_new, c, a)
            
            if OutputFlag == 1:
                print('Cluster splitted', c,'| Action causing contradiction:', a, '| Cluster most elements went to:', b)
            df_new = split(df_new, c, a, b, pfeatures, nc, classification,split_classifier_params)
            
            # error and accuracy calculations
            
            R2_train = R2_value_training(df_new)
            training_R2.append(R2_train)       
            train_error = training_value_error(df_new, relative=False, h=h)
            training_error.append(train_error)            
            
            if grid:
                train_acc = training_accuracy(df_new)[0]
                training_acc.append(train_acc)                        
            
            
            if testing:
                model = predict_cluster(df_new, pfeatures)
                R2_test = R2_value_testing(df_test, df_new, model, pfeatures)
                testing_R2.append(R2_test)
                test_error = testing_value_error(df_test, df_new, model, pfeatures,relative=False,h=h)
                testing_error.append(test_error)
                
                if grid:
                    test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
                    testing_acc.append(test_acc)
                
                
            
            # printing error and accuracy values
            if OutputFlag == 1:
                print('training value R2:', R2_train)
                print('training value error:', train_error)
                if grid:
                    print('training accuracy:', train_acc)
                if testing:
                    print('testing value R2:', R2_test)
                    print('testing value error:', test_error)
                    if grid:
                        print('testing accuracy:', test_acc)
            #print('predictions:', get_predictions(df_new))
            #print(df_new.head())
            cont = True
            nc += 1
        if not cont:
            break
        if nc >= max_k:
            print('Optimal # of clusters reached')
            break
    if OutputFlag == 1:
        print(df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count())
    
    
    # plotting functions
    ## Plotting accuracy and value R2
    its = np.arange(k+1, nc+1)
    if plot:
        if grid:
            fig1, ax1 = plt.subplots()
            ax1.plot(its, training_R2, label= "Training R2")
            ax1.plot(its, training_acc, label = "Training Accuracy")
            if testing:
                ax1.plot(its, testing_acc, label = "Testing Accuracy")
                ax1.plot(its, testing_R2, label = "Testing R2")
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
        if testing:
            ax2.plot(its, testing_error, label = "Testing Error")
        if n>0:
            ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        ax2.set_ylim(0)
        ax2.set_xlabel('# of Clusters')
        ax2.set_ylabel('Value error')
        ax2.set_title('Value error by number of clusters')
        ax2.legend()
        plt.show()
    
    df_train_error = pd.DataFrame(list(zip(its, training_error)), \
                                  columns = ['Clusters', 'Error'])
    if testing:
        df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
                                  columns = ['Clusters', 'Error'])
        return (df_new,df_train_error,df_test_error)
    
    return(df_new,df_train_error,testing_error)

#################################################################



# Splitter algorithm with cross-validation
def fit_CV(df,
          pfeatures,
          th,
          clustering,
          distance_threshold,
          classification,
          split_classifier_params,
          max_k,
          n_clusters,
          random_state,
          h,
          OutputFlag = 0,
          cv=5,
          n=-1,
          plot = False):
    
    
    df_training_error = pd.DataFrame(columns=['Clusters'])
    df_testing_error = pd.DataFrame(columns=['Clusters'])
    
    gkf = GroupKFold(n_splits=cv)
    # shuffle the ID's (create a new column), and do splits based on new ID's
    random.seed(datetime.now())
    g = [df for _, df in df.groupby('ID')]
    random.shuffle(g)
    df = pd.concat(g).reset_index(drop=True)
    ids = df.groupby(['ID'], sort=False).ngroup()
    df['ID_shuffle'] = ids
    for train_idx, test_idx in gkf.split(df, y=None, groups=df['ID_shuffle']):

        df_train = df[df.index.isin(train_idx)]
        df_test = df[df.index.isin(test_idx)]
        #print('IDs in testing', df_test['ID'].unique())
        #################################################################
        # Initialize Clusters
        df_init = initializeClusters(df_train,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                distance_threshold = distance_threshold,
                                random_state=random_state)
        k = df_init['CLUSTER'].nunique()
        #print('k', k)
        #print(df_init)
        #################################################################
        
        #################################################################
        # Run Iterative Learning Algorithm
        
        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures,
                                          th,
                                          df_test,
                                          testing = True,
                                          max_k = max_k,
                                          classification = classification,
                                          split_classifier_params = split_classifier_params,
                                          h = h, 
                                          OutputFlag = 0,
                                          n=n,
                                          plot = plot)
        
        df_training_error = df_training_error.merge(training_error, \
                                                    how='outer', on=['Clusters'])
        df_testing_error = df_testing_error.merge(testing_error, \
                                                  how='outer', on=['Clusters'])
    
    df_training_error.set_index('Clusters', inplace=True)
    df_testing_error.set_index('Clusters', inplace=True)
    df_training_error.dropna(inplace=True)
    df_testing_error.dropna(inplace=True)
    #print(df_training_error)
    #print(df_testing_error)
    cv_training_error = np.mean(df_training_error, axis=1)
    cv_testing_error = np.mean(df_testing_error, axis=1)
    #print(cv_training_error)
    #print(cv_testing_error)
    
    
    if plot:
        fig1, ax1 = plt.subplots()
        #its = np.arange(k+1,k+1+len(cv_training_error))
        ax1.plot(cv_training_error.index.values, cv_training_error, label= "CV Training Error")
        #ax1.plot(its, cv_testing_error, label = "CV Testing Error")
        ax1.plot(cv_testing_error.index.values, cv_testing_error, label= "CV Testing Error")
        #ax1.plot(its, training_acc, label = "Training Accuracy")
        #ax1.plot(its, testing_acc, label = "Testing Accuracy")
        if n>0:
            ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        ax1.set_ylim(0)
        ax1.set_xlabel('# of Clusters')
        ax1.set_ylabel('Mean CV Error or Accuracy %')
        ax1.set_title('Mean CV Error and Accuracy During Splitting')
        ax1.legend()
    
    return (cv_training_error,cv_testing_error)