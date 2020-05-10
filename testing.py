# -*- coding: utf-8 -*-
"""
This file is intended in getting several test measure on the output of the algorithm
Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import binascii
from sklearn.tree import DecisionTreeClassifier

#maps each OG_ClUSTER to a cluster --> We map a cluster to the OG_CLUSTER most 
#present in it
def get_predictions(df_new):
    df0 = df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count()
    df0 = df0.groupby('CLUSTER').idxmax()
    df2 = pd.DataFrame()
    df2['OG_CLUSTER'] = df0.apply(lambda x: x[1])
    return df2



#Returns the global training accuracy and a df of training accuracy per OG_CLUSTER
def training_accuracy(df_new):
    clusters = get_predictions(df_new)
    print('Clusters', clusters)
    #First term is what the algo predicts for each training data points, sets 
    #term is what is the truth
    accuracy = clusters.loc[df_new['CLUSTER']].reset_index()['OG_CLUSTER'] == df_new.reset_index()['OG_CLUSTER']
    #print(accuracy)
    #accuracy = clusters.loc[df_new['CLUSTER'] == df_new['OG_CLUSTER']]
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_new.reset_index()['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy,accuracy_df)

#Returns the testing accuracy of 
def testing_accuracy(df_test, df_new, model, pfeatures):
    clusters = get_predictions(df_new)
    
    test_clusters = model.predict(df_test.iloc[:, 2:2+pfeatures])
    df_test['CLUSTER'] = test_clusters
    
    accuracy = clusters.loc[df_test['CLUSTER']].reset_index()['OG_CLUSTER'] == df_test.reset_index()['OG_CLUSTER']
    #print(accuracy)
    #accuracy = clusters.loc[df_new['CLUSTER'] == df_new['OG_CLUSTER']]
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_test.reset_index()['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy,accuracy_df)
    

#Get estimated MDP from clustering: for a given cluster s and action a, the next 
# cluster is the one we go most to in the data when being in s and taking a
def get_MDP(df_new):
    #removing None values when counting where clusteres go
    df0 = df_new[df_new['NEXT_CLUSTER']!='None']
    transition_df = df0.groupby(['CLUSTER','ACTION','NEXT_CLUSTER'])['FEATURE_1'].count()
    #taking the target cluster where we went the most
    transition_df = transition_df.groupby(['CLUSTER','ACTION']).idxmax()
    P_df = pd.DataFrame()
    P_df['NEXT_CLUSTER'] = transition_df.apply(lambda x: x[2])
    R_df = df_new.groupby('CLUSTER')['RISK'].mean()
    return P_df,R_df

#predicts value given a cluster and actions
def predict_value_of_cluster(P_df,R_df,cluster,actions):
    s = cluster
    v = R_df.loc[s]
    for a in actions:
        s = P_df.loc[s,a].values[0]
        v = v + R_df.loc[s]
    return v

#Compute E((\hat{v}-v)^2) ie the expect error in estimating the value given actions
#NEED TO CHANGE: must adapt to case where T is diffrent from one simulation to another.
# add Ids of simulations
def training_value_error(df_new):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
   # print(df2)
    N = df2.shape[0]
    #print(df2)
    #print(df_new)
    for i in range(N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]
        #print('s, a, r', s, a, v_true)
        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        #print('index', index, index.dtype)
        #index = int(index)
        cont = True
        t = 1
        while cont:
            #print('index', index, 't', t)
            #print('next risk', df_new['RISK'].loc[index + t])
            v_true = v_true + df_new['RISK'].loc[index + t]
            #print('new v_true', v_true)
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[index + t]
            #print('new action', a, ' \n')
            #print('next estimated risk', R_df.loc[s])
            v_estim = v_estim + R_df.loc[s]
            try: 
                df_new['ID'].loc[index+t+1]
            except:
                break
            if df_new['ID'].loc[index+t] != df_new['ID'].loc[index+t+1]:
                break
            t += 1
        #print('true vs estimate', v_true, v_estim)
        E_v = E_v + (v_true-v_estim)**2
    return (E_v/N)
#    return np.sqrt((E_v/N))/(R_df.max()*T)

def training_value_error_old(df_new,N,T):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    for i in range(N):
        s = df_new['CLUSTER'].loc[i*T]
        a = df_new['ACTION'].loc[i*T]
        v_true = df_new['RISK'].loc[i*T]
        #print('s, a, v_true', s, a, v_true)
        v_estim = R_df.loc[s]
        for t in range(1,T):
            v_true = v_true + df_new['RISK'].loc[i*T+t]
            #print('new v_true', v_true)
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[i*T+t]
            #print('new action', a)
            v_estim = v_estim + R_df.loc[s]
        E_v = E_v + (v_true-v_estim)**2
    return (E_v/N)

# takes in a dataframe of testing data, and dataframe of new clustered data
# predicts initial cluster, and then run analysis on v_predict and v_true
def testing_value_error(df_test, df_new, model, pfeatures):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    
    clusters = model.predict(df2.iloc[:, 2:2+pfeatures])
    df2['CLUSTER'] = clusters
    #print(df2)
    for i in range (N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]
        #print('s, a, r', s, a, v_true)
        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        while cont:
            #print('index', index, 't', t)
            #print('next risk', df_test['RISK'].loc[index + t])
            v_true = v_true + df_test['RISK'].loc[index + t]
            #print('new v_true', v_true)
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test['ACTION'].loc[index + t]
            #print('new action', a, ' \n')
            v_estim = v_estim + R_df.loc[s]
            #print('next estimated risk', R_df.loc[s])
            try: 
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break
            t += 1
        #print('true vs estimate', v_true, v_estim)
        E_v = E_v + (v_true-v_estim)**2
    return (E_v/N)

# function that takes df_new and pfeatures, and returns a prediction model m
def predict_cluster(df_new, pfeatures):
    X = df_new.iloc[:, 2:2+pfeatures]
    y = df_new['CLUSTER']
    m = DecisionTreeClassifier()
    m.fit(X, y)
    return m
    


def R2_value_training(df_new):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    V_true = []
    for i in range(N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]
        V_true.append(v_true)
        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        while cont:
            v_true = v_true + df_new['RISK'].loc[index + t]
            V_true.append(v_true)

            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[index + t]

            v_estim = v_estim + R_df.loc[s]
            try: 
                df_new['ID'].loc[index+t+1]
            except:
                break
            if df_new['ID'].loc[index+t] != df_new['ID'].loc[index+t+1]:
                break
            t += 1
        E_v = E_v + (v_true-v_estim)**2
    E_v = E_v/N
    #print('new E_v', E_v)
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    #print('new v_mean', v_mean)
    SS_tot = sum((V_true-v_mean)**2)/N
    return 1- E_v/SS_tot

def R2_value_testing(df_test, df_new, model, pfeatures):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    
    clusters = model.predict(df2.iloc[:, 2:2+pfeatures])
    df2['CLUSTER'] = clusters
    
    V_true = []
    for i in range(N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]
        V_true.append(v_true)
        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        while cont:
            v_true = v_true + df_test['RISK'].loc[index + t]
            V_true.append(v_true)

            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test['ACTION'].loc[index + t]

            v_estim = v_estim + R_df.loc[s]
            try: 
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break
            t += 1
        E_v = E_v + (v_true-v_estim)**2
    E_v = E_v/N
    #print('new E_v', E_v)
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    #print('new v_mean', v_mean)
    SS_tot = sum((V_true-v_mean)**2)/N
    return 1- E_v/SS_tot
    

#Computes the R square of value prediction
def R2_value_old(df_new,N,T):
    P_df,R_df = get_MDP(df_new)
    E_v = 0
    V_true = []
    for i in range(N):
        s = df_new['CLUSTER'].loc[i*T]
        a = df_new['ACTION'].loc[i*T]
        v_true = df_new['RISK'].loc[i*T]
        V_true.append(v_true)
        v_estim = R_df.loc[s]
        for t in range(1,T):
            v_true = v_true + df_new['RISK'].loc[i*T+t]
            V_true.append(v_true)
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[i*T+t]
            v_estim = v_estim + R_df.loc[s]
        E_v = E_v + (v_true-v_estim)**2
    E_v = E_v/N
    print('new E_v', E_v)
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    print('new v_mean', v_mean)
    SS_tot = sum((V_true-v_mean)**2)/N
    return 1- E_v/SS_tot
# Returns the purity of each cluster
def Purity(df):
    su = pd.DataFrame(df.groupby(['CLUSTER'])['OG_CLUSTER']
    .value_counts(normalize=True)).reset_index(level=0)
    su.columns= ['CLUSTER','Purity']
    return su.groupby('CLUSTER')['Purity'].max()



def plot_features(df):
    df.plot.scatter(x='FEATURE_1',
                      y='FEATURE_2',
                      c='OG_CLUSTER',
                      colormap='viridis')
    plt.show()


#code for splitting data into training & testing based on ID
def test_set_check(identifier, test_ratio):
    return binascii.crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

#returns Testing and Training dataset
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

