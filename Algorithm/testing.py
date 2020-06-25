# -*- coding: utf-8 -*-
"""
This file is intended to perform various testing measurements on the output of 

the MDP Clustering Algorithm. 

Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""
#################################################################
# Load Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
#################################################################


#################################################################
# Functions for Predictions

# predict_cluster() takes in a clustered dataframe df_new, the number of 
# features pfeatures, and returns a prediction model m that predicts the most
# likely cluster from a datapoint's features
def predict_cluster(df_new, # dataframe: trained clusters
                    pfeatures): # int: # of features
    X = df_new.iloc[:, 2:2+pfeatures]
    y = df_new['CLUSTER']
    
    params = {
    'max_depth': [3, 4, 6, 10,None]
    }

    m = DecisionTreeClassifier()
    
    m = GridSearchCV(m, params,cv = 5, iid=True) #will return warning if 'iid' param not set to true

#    m = DecisionTreeClassifier(max_depth = 10)
    m.fit(X, y)
    return m


# predict_value_of_cluster() takes in MDP parameters, a cluster label, and 
# and a list of actions, and returns the predicted value of the given cluster
# currently takes value of current cluster as well as next cluster
def predict_value_of_cluster(P_df,R_df, # df: MDP parameters
                             cluster, # int: cluster number
                             actions): # list: list of actions
    s = cluster
    v = R_df.loc[s]
    for a in actions:
        s = P_df.loc[s,a].values[0]
        v = v + R_df.loc[s]
    return v


# get_MDP() takes in a clustered dataframe df_new, and returns dataframes  
# P_df and R_df that represent the parameters of the estimated MDP
def get_MDP(df_new):
    # removing None values when counting where clusters go
    df0 = df_new[df_new['NEXT_CLUSTER']!='None']
    #df0 = df_new
    transition_df = df0.groupby(['CLUSTER','ACTION','NEXT_CLUSTER'])['RISK'].count()

    # next cluster given how most datapionts transition for the given action
    transition_df = transition_df.groupby(['CLUSTER','ACTION']).idxmax()
    P_df = pd.DataFrame()
    P_df['NEXT_CLUSTER'] = transition_df.apply(lambda x: x[2])
    R_df = df_new.groupby('CLUSTER')['RISK'].mean()
    return P_df,R_df
#################################################################

    

#################################################################
# Functions for Error 

# training_value_error() takes in a clustered dataframe, and computes the 
# E((\hat{v}-v)^2) expected error in estimating values (risk) given actions
# Returns a float of average value error per ID 
def training_value_error(df_new, #Outpul of algorithm
                         relative=False, #Output Raw error or RMSE ie ((\hat{v}-v)/v)^2
                         h=5): # Length of forecast. The error is computed on v_h = \sum_{t=h}^H v_t
                               # if h = -1, we forecast the whole path
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_train = df2.shape[0]
    

    for i in range(N_train):
        index = df2['index'].iloc[i]
        # initializing first state for each ID
        cont = True
        
        if h == -1:
            t = 0
            
        else:
            H = -1
                # Computing Horizon H of ID i
            while cont:
                H+= 1
                try: 
                    df_new['ID'].loc[index+H+1]
                except:
                    break
                if df_new['ID'].loc[index+H] != df_new['ID'].loc[index+H+1]:
                    break
            t = H-h
            
        v_true = 0
        v_estim = 0
        s = df_new['CLUSTER'].loc[index + t]
        a = df_new['ACTION'].loc[index + t]
        
        # predicting path of each ID
        while cont:
            #if a == 'None':
                #break
            v_true = v_true + df_new['RISK'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
            
            try: 
                df_new['ID'].loc[index+t+1]
            except:
                break
            if df_new['ID'].loc[index+t] != df_new['ID'].loc[index+t+1]:
                break
            
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
           
            t += 1
            a = df_new['ACTION'].loc[index + t]
        if relative:
            E_v = E_v + ((v_true-v_estim)/v_true)**2
        else:
            E_v = E_v + (v_true-v_estim)**2
    E_v = (E_v/N_train)
    return np.sqrt(E_v)


# testing_value_error() takes in a dataframe of testing data, and dataframe of 
# new clustered data, a model from predict_cluster function, and computes the
# expected value error given actions and a predicted initial cluster and time
# horizon h (ifh = -1, we forecast the whole path)
# Returns a float of sqrt average value error per ID
def testing_value_error(df_test, df_new, model, pfeatures,relative=False,h=5):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N_test = df2.shape[0]
        
    df_test['CLUSTER'] = model.predict(df_test.iloc[:, 2:2+pfeatures])
    for i in range(N_test):
        # initializing index of first state for each ID
        index = df2['index'].iloc[i]
        cont = True
        
        if h == -1:
            t = 0

        else:
            H = -1
            # Computing Horizon H of ID i
            while cont:
                H+= 1
                try: 
                    df_test['ID'].loc[index+H+1]
                except:
                    break
                if df_test['ID'].loc[index+H] != df_test['ID'].loc[index+H+1]:
                    break
            t = H-h
        
        v_true = 0
        v_estim = 0
        s = df_test['CLUSTER'].loc[index + t]
        a = df_test['ACTION'].loc[index + t]
        
        # predicting path of each ID
        while cont:
            v_true = v_true + df_test['RISK'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            
            try: 
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break 
            t += 1
            a = df_test['ACTION'].loc[index + t]
        if relative:
            E_v = E_v + ((v_true-v_estim)/v_true)**2
        else:
            E_v = E_v + (v_true-v_estim)**2

    E_v = (E_v/N_test)
    return np.sqrt(E_v)
    
#################################################################
    

#################################################################
# Functions for R2 Values

# R2_value_training() takes in a clustered dataframe, and returns a float 
# of the R-squared value between the expected value and true value of samples
# currently doesn't support horizon h specifications
def R2_value_training(df_new):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    #print(P_df)
    df2 = df_new.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    V_true = []
    for i in range(N):
        # initializing starting cluster and values
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]

        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        # iterating through path of ID
        while cont:
            v_true = v_true + df_new['RISK'].loc[index + t]
            try: 
                df_new['ID'].loc[index+t+1]
            except:
                break
            if df_new['ID'].loc[index+t] != df_new['ID'].loc[index+t+1]:
                break
            
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            except TypeError:
                print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new['ACTION'].loc[index + t]
            v_estim = v_estim + R_df.loc[s]
            
            t += 1
        E_v = E_v + (v_true-v_estim)**2
        V_true.append(v_true)
    # defining R2 baseline & calculating the value
    E_v = E_v/N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true-v_mean)**2)/N
    return max(1- E_v/SS_tot,0)


# R2_value_testing() takes a dataframe of testing data, a clustered dataframe, 
# a model outputted by predict_cluster, and returns a float of the R-squared
# value between the expected value and true value of samples in the test set
# currently doesn't support horizon h specifications
def R2_value_testing(df_test, df_new, model, pfeatures):
    E_v = 0
    P_df,R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(['ID']).first()
    N = df2.shape[0]
    
    # predicting clusters based on features
    clusters = model.predict(df2.iloc[:, 2:2+pfeatures])
    df2['CLUSTER'] = clusters
    
    V_true = []
    for i in range(N):
        s = df2['CLUSTER'].iloc[i]
        a = df2['ACTION'].iloc[i]
        v_true = df2['RISK'].iloc[i]

        v_estim = R_df.loc[s]
        index = df2['index'].iloc[i]
        cont = True
        t = 1
        while cont:
            v_true = v_true + df_test['RISK'].loc[index + t]
            

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
        V_true.append(v_true)
    E_v = E_v/N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true-v_mean)**2)/N
    return max(1- E_v/SS_tot,0)
#################################################################


#################################################################
# Functions for Plotting and Visualization
    
# plot_features() takes in a dataframe and two features, and plots the data
# to illustrate the noise in each cluster
def plot_features(df, x, y, c='CLUSTER'):
    df.plot.scatter(x=x,
                      y=y,
                      c=c,
                      colormap='viridis')
#    import seaborn as sns
#    sns.pairplot(x_vars=["FEATURE_1"], y_vars=["FEATURE_2"], data=df, hue="OG_CLUSTER", height=5)
    plt.show()

# cluster_size() takes a dataframe, and returns the main statistics of each
# cluster in a dataframe
def cluster_size(df):
    df2 = df.groupby('CLUSTER')['RISK'].agg(['count','mean','std','min','max'])
    df2['rel'] = 100*abs(df2['std']/df2['mean'])
    df2['rel_mean'] = 100*abs(df2['std']/df['RISK'].mean())
    return df2


# next_clusters() takes a dataframe, and returns a chart showing transitions from
# each cluster/action pair, and the purity of the highest next_cluster. 
# Disregards those with 'NEXT_CLUSTER' = None, and returns a dataframe of the results
def next_clusters(df):
    df = df.loc[df['NEXT_CLUSTER']!='None']
    df2 = df.groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])['RISK'].agg(['count'])
    df2['purity'] = df2['count']/df.groupby(['CLUSTER', 'ACTION'])['RISK'].count()
    df2.reset_index(inplace=True)
    idx = df2.groupby(['CLUSTER', 'ACTION'])['count'].transform(max) == df2['count']
    df_final = df2[idx].groupby(['CLUSTER','ACTION']).max()
    return df_final


# decision_tree() takes in a trained MDP model, outputs a pdf of 
# the best decision tree, as well as other visualizations
def decision_tree(model):
    # assumes that m.m, the prediction model, is a GridSearchCV object
    dc = model.m.best_estimator_
    
    # creating the decision tree diagram in pdf: 
    dot_data = tree.export_graphviz(dc, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render("Decision_Tree")
    return graph
    
    
    
#################################################################
    

#################################################################
# Functions for Grid Testing (Predictions, Accuracy, Purity)
    
# get_predictions() takes in a clustered dataframe df_new, and maps each 
# CLUSTER to an OG_CLUSTER that has the most elements
# Returns a dataframe of the mappings
def get_predictions(df_new):
    df0 = df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count()
    df0 = df0.groupby('CLUSTER').idxmax()
    df2 = pd.DataFrame()
    df2['OG_CLUSTER'] = df0.apply(lambda x: x[1])
    return df2
    

# training_accuracy() takes in a clustered dataframe df_new, and returns the 
# average training accuracy of all clusters (float) and a dataframe of 
# training accuracies for each OG_CLUSTER
def training_accuracy(df_new):
    clusters = get_predictions(df_new)
#    print('Clusters', clusters)
    
    # Tallies datapoints where the algorithm correctly classified a datapoint's
    # original cluster to be the OG_CLUSTER mapping of its current cluster
    accuracy = clusters.loc[df_new['CLUSTER']].reset_index()['OG_CLUSTER'] \
                                        == df_new.reset_index()['OG_CLUSTER']
    #print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_new.reset_index()['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy, accuracy_df)


# testing_accuracy() takes in a testing dataframe df_test (unclustered), 
# a df_new clustered dataset, a model from predict_cluster and 
# Returns a float for the testing accuracy measuring how well the model places
# testing data into the right cluster (mapped from OG_CLUSTER), and 
# also returns a dataframe that has testing accuracies for each OG_CLUSTER
def testing_accuracy(df_test, # dataframe: testing data
                     df_new, # dataframe: clustered on training data
                     model, # function: output of predict_cluster
                     pfeatures): # int: # of features
    
    clusters = get_predictions(df_new)
    
    test_clusters = model.predict(df_test.iloc[:, 2:2+pfeatures])
    df_test['CLUSTER'] = test_clusters
    
    accuracy = clusters.loc[df_test['CLUSTER']].reset_index()['OG_CLUSTER'] \
                                        == df_test.reset_index()['OG_CLUSTER']
    #print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_test.reset_index()['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy, accuracy_df)


# purity() takes a clustered dataframe and returns a dataframe with the purity 
# of each cluster
def purity(df):
    su = pd.DataFrame(df.groupby(['CLUSTER'])['OG_CLUSTER']
    .value_counts(normalize=True)).reset_index(level=0)
    su.columns= ['CLUSTER','Purity']
    return su.groupby('CLUSTER')['Purity'].max()
#################################################################

