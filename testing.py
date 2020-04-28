# -*- coding: utf-8 -*-
"""
This file is intended in getting several test measure on the output of the algorithm
Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    #First term is what the algo predicts for each training data points, sets 
    #term is what is the truth
    accuracy = clusters.loc[df_new['CLUSTER']].reset_index()['OG_CLUSTER'] == df_new['OG_CLUSTER']
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_new['OG_CLUSTER']
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
def traning_value_error(df_new,N,T):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    for i in range(N):
        s = df_new['CLUSTER'].loc[i*T]
        a = df_new['ACTION'].loc[i*T]
        v_true = df_new['RISK'].loc[i*T]
        v_estim = R_df.loc[s]
        for t in range(1,T):
            v_true = v_true + df_new['RISK'].loc[i*T+t]
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[i*T+t]
            v_estim = v_estim + R_df.loc[s]
        E_v = E_v + (v_true-v_estim)**2
    return (E_v/N)
#    return np.sqrt((E_v/N))/(R_df.max()*T)
        
#Computes the R square of value prediction
def R2_value(df_new,N,T):
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
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true-v_mean)**2)/N
    return 1- E_v/SS_tot
# Returns the purity of each cluster
def Purity(df):
    su = pd.DataFrame(df.groupby(['CLUSTER'])['OG_CLUSTER']
    .value_counts(normalize=True)).reset_index(level=0)
    su.columns= ['CLUSTER','Purity']
    return su.groupby('CLUSTER')['Purity'].max()



def plot_features(df):
    x=  list(df['FEATURE_1'])
    y=  list(df['FEATURE_2'])
    plt.scatter(x, y)
    plt.show()