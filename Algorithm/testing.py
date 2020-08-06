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
import random
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
# P_df and R_df that represent the parameters of the estimated MDP (if sink
# exists, it will be the last cluster and goes to itself)
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
    
    # check if end state exists, if so make a sink node
    if 'End' in P_df['NEXT_CLUSTER'].unique():
        P_df = P_df.reset_index()
        
        # find goal cluster that leads to sink, then remove
        c = P_df.loc[P_df['NEXT_CLUSTER']=='End']['CLUSTER'].max()
        P_df = P_df.loc[P_df['NEXT_CLUSTER']!='End']
        
        # create dataframe that goal go to sink and sink go to sink
        s = P_df['CLUSTER'].max() + 1
        actions = P_df['ACTION'].unique()
        df_end = []
        for a in actions:
            df_end.append([c, a, s])
            df_end.append([s, a, s])
        df_end = pd.DataFrame(df_end, columns = ['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])
        
        P_df = P_df.append(df_end)
        P_df.sort_values(by=['CLUSTER','ACTION'], inplace=True)
        P_df.set_index(['CLUSTER','ACTION'], inplace=True)
        
        # set new reward node 
        R_df = R_df.append(pd.Series([0], index=[s]))
        
        # print "end state defined as cluster __" 
        
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
            #except ValueError
            except:
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
                df_test['ID'].loc[index+t+1]
            except:
                break
            if df_test['ID'].loc[index+t] != df_test['ID'].loc[index+t+1]:
                break 
            
            try:
                s = P_df.loc[s,a].values[0]
            # error raises in case we never saw a given transition in the data
            
            #except TypeError: # sometimes we see KeyError or IndexError...
            except:
                print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            
            
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
            #except TypeError:
            except:
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
            #except TypeError:
            except:
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
                      colormap='tab20')
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
# each cluster/action pair, count of each cluster/action pair, and the purity 
# of the highest next_cluster. 
# Disregards those with 'NEXT_CLUSTER' = None, and returns a dataframe of the results
def next_clusters(df):
    df = df.loc[df['NEXT_CLUSTER']!='None']
    df2 = df.groupby(['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])['RISK'].agg(['count'])
    df2['purity'] = df2['count']/df.groupby(['CLUSTER', 'ACTION'])['RISK'].count()
    df2.reset_index(inplace=True)
    idx = df2.groupby(['CLUSTER', 'ACTION'])['count'].transform(max) == df2['count']
    df_final = df2[idx].groupby(['CLUSTER','ACTION']).max()
    df_final['count'] = df2.groupby(['CLUSTER', 'ACTION'])['count'].sum()
    return df_final


# decision_tree_diagram() takes in a trained MDP model, outputs a pdf of 
# the best decision tree, as well as other visualizations
def decision_tree_diagram(model):
    # assumes that m.m, the prediction model, is a GridSearchCV object
    dc = model.m.best_estimator_
    
    # creating the decision tree diagram in pdf: 
    dot_data = tree.export_graphviz(dc, out_file=None,
                                    filled=True, rounded=True,
                                    special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render("Decision_Tree_Diagram")

    return graph


# decision_tree_regions() takes a model and plots a visualization of the
# decision regions of two of the features (currently first and second)
def decision_tree_regions(model):
    dc = model.m.best_estimator_
    n_classes = model.df_trained['CLUSTER'].max()
    plot_step = 0.02
    
    plt.subplot()
    x_min = model.df_trained.iloc[:, 2].min() - 1 
    x_max = model.df_trained.iloc[:, 2].max() + 1 
    y_min = model.df_trained.iloc[:, 3].min() - 1
    y_max = model.df_trained.iloc[:, 3].max() + 1 
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = dc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    for i in range(n_classes):
        idx = np.where(model.df_trained['CLUSTER'] == i)
        
        r = random.random()
        b = random.random()
        g = random.random()
        color = np.array([[r, g, b]])
        #colors = ['r', 'y', 'b']
        #color = colors[i%3]
        plt.scatter(model.df_trained.iloc[idx].iloc[:, 2], \
                    model.df_trained.iloc[idx].iloc[:, 3], c=color,
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        
    plt.show()
    return

# NOT TESTED YET! TEST ON HIV WHEN MODEL TRAINED!
# model_trajectory() takes a trained model, the real transition function of
# the model f(x, u), the initial state x, and plots how the model's optimal 
# policy looks like on the start state according to f1 and f2 two features 
# indices e.g. x[f1] x[f2] plotted on the x and y axes, for n steps
def model_trajectory(m, 
                    f, 
                    x, 
                    f1=0, 
                    f2=1, 
                    n=50):
    if m.v is None:
        m.solve_MDP()
    
    xs = [x[f1]]
    ys = [x[f2]]

    for i in range(n):
        # find current state and action
        s = m.m.predict(np.array(x).reshape(1, -1))
        #print(s)
        a = int(m.pi[s])
        #print(a)
        x_new = f(x, a)
        if x_new[0] == None:
            break
        
        xs.append(x_new[f1])
        ys.append(x_new[f2])
        x = x_new
    
    # TODO: not plot the sink
    xs = np.array(xs)
    ys = np.array(ys)
    
    
    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u/2
    pos_y = ys[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    
    fig, ax = plt.subplots()
    ax.plot(xs,ys, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    #ax.set_xlabel('FEATURE_%i' %f1)
    #ax.set_ylabel('FEATURE_%i' %f2)
    
    # set plot limits if relevant
    #plt.ylim(-l+0.5, 0.5)
    #plt.xlim(-.5, l-0.5)
    plt.show()
    return xs, ys
    
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
    
    test_clusters = model.m.predict(df_test.iloc[:, 2:2+pfeatures])
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


# generalization_accuracy() plots the training and testing accuracies as above
# for a given list of models and a test-set.
def generalization_accuracy(models, df_test, Ns):
    tr_accs = []
    test_accs = []
    for model in models:
        tr_acc, df = training_accuracy(model.df_trained)
        tr_accs.append(tr_acc)
        
        test_acc, df_t = testing_accuracy(df_test, model.df_trained, model, model.pfeatures)
        test_accs.append(test_acc)
    
    fig1, ax1 = plt.subplots()
    ax1.plot(Ns, tr_accs, label = 'Training Accuracy')
    ax1.plot(Ns, test_accs, label = 'Testing Accuracy')
    ax1.set_xlabel('N training data size')
    ax1.set_ylabel('Accuracy %')
    ax1.set_title('Model Generalization Accuracies')
    plt.legend()
    plt.show()
    return
    
    
#################################################################

