#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:47:03 2020

Model Class that runs the Iterative Clustering algorithm on any data.  

@author: janiceyang
"""

#################################################################
# Load Libraries
import pandas as pd
import numpy as np

from clustering import fit_CV, initializeClusters, splitter
from testing import predict_cluster, training_value_error, get_MDP, \
        predict_value_of_cluster, testing_value_error, model_trajectory, \
        next_clusters
from MDPtools import SolveMDP
from sklearn.metrics import accuracy_score
#################################################################

class MDP_model:
    def __init__(self):
        self.df = None # original dataframe from data
        self.pfeatures = None # number of features
        self.CV_error = None # error at minimum point of CV
        self.CV_error_all = None # errors of different clusters after CV
        self.training_error = None # training errors after last split sequence
        self.opt_k = None # number of clusters in optimal clustering
        self.df_trained = None # dataframe after optimal training
        self.m = None # model for predicting cluster number from features
        self.clus_pred_accuracy = None # accuracy score of the cluster prediction function
        self.P_df = None # Transition function of the learnt MDP
        self.R_df = None # Reward function of the learnt MDP
        self.nc = None # dataframe similar to P_df, but also includes 'count' and 'purity' cols
        self.v = None # value after MDP solved
        self.pi = None # policy after MDP solved
        self.P = None # P_df but in matrix form of P[a, s, s']
        self.R = None # R_df but in matrix form of R[a, s]
        
        
    # fit_CV() takes in parameters for prediction, and trains the model on the 
    # optimal clustering for a given horizon h (# of actions), using cross
    # validation.
    def fit_CV(self, 
            data, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
            pfeatures, # int: number of features
            h=5, # int: time horizon (# of actions we want to optimize)
            max_k=70, # int: max number of clusters
            distance_threshold = 0.05, # clustering diameter for Agglomerative clustering
            cv=5, # number for cross validation
            th=0, # splitting threshold
            classification = 'DecisionTreeClassifier', # classification method
            split_classifier_params = {'random_state': 0},
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0,
            plot = False):
        
        df = data.copy()
            
        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        
        # run cross validation on the data to find best clusters
        cv_training_error,cv_testing_error =fit_CV(self.df,
                                                  self.pfeatures,
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
                                                  cv=cv,
                                                  n=-1,
                                                  plot = plot)
        
        # find the best cluster
        k = cv_testing_error.idxmin()
        print('best clusters:', k)
        
        # save total error and error corresponding to chosen model
        self.CV_error = cv_testing_error.loc[k]
        self.CV_error_all = cv_testing_error
        self.opt_k = k
        
        # actual training on all the data
        df_init = initializeClusters(self.df,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                distance_threshold = distance_threshold,
                                random_state=random_state)
        
        # change end state to 'end'
        df_init.loc[df_init['ACTION']=='None', 'NEXT_CLUSTER'] = 'End'
        
        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = self.opt_k,
                                          classification=classification,
                                          split_classifier_params = split_classifier_params,
                                          h=h,
                                          OutputFlag = 0,
                                          plot = plot)
        
        
        # storing trained dataset and predict_cluster function and accuracy
        self.df_trained = df_new
        self.m = predict_cluster(df_new, self.pfeatures)
        pred = self.m.predict(df_new.iloc[:, 2:2+self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, df_new['CLUSTER'])
        
        # store final training error
        self.training_error = training_value_error(self.df_trained)
        
        
        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df
        
        # store next_clusters dataframe
        self.nc = next_clusters(df_new)
        
    
    # fit() takes in the parameters for prediction, and directly fits the model
    # to the data without running cross validation
    def fit(self, 
            data, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
            pfeatures, # int: number of features
            h=5, # int: time horizon (# of actions we want to optimize)
            max_k=70, # int: max number of clusters
            distance_threshold = 0.05, # clustering diameter for Agglomerative clustering
            cv=5, # number for cross validation
            th=0, # splitting threshold
            classification = 'DecisionTreeClassifier', # classification method
            split_classifier_params = {'random_state':0}, # dict of classifier params
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0,
            plot = False,
            optimize = True):
    
        df = data.copy()
            
        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        
        # training on all the data
        df_init = initializeClusters(self.df,
                                clustering=clustering,
                                n_clusters=n_clusters,
                                distance_threshold = distance_threshold,
                                random_state=random_state)
        
        # change end state to 'end'
        df_init.loc[df_init['ACTION']=='None', 'NEXT_CLUSTER'] = 'End'
        print('Clusters Initialized')
        print(df_init)
        
        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = max_k,
                                          classification=classification,
                                          split_classifier_params = split_classifier_params,
                                          h=h,
                                          OutputFlag = 0,
                                          plot = plot)
        
        # store all training errors
        self.training_error = training_error
        
        # if optimize, find best cluster and resplit
        if optimize: 
            k = self.training_error['Clusters'].iloc[self.training_error['Error'].idxmin()]
            for i in range(k):
                # if clustering is less than k but error within 10^-14, take this
                if abs(self.training_error['Error'].min() - \
                       self.training_error.loc[self.training_error['Clusters']==i]['Error'].min()) < 1e-14:
                    k = i
                    break
            self.opt_k = k
            df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = self.opt_k,
                                          classification=classification,
                                          split_classifier_params = split_classifier_params,
                                          h=h,
                                          OutputFlag = 0,
                                          plot = plot)
        
        # storing trained dataset and predict_cluster function
        self.df_trained = df_new
        self.m = predict_cluster(df_new, self.pfeatures)
        pred = self.m.predict(df_new.iloc[:, 2:2+self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, df_new['CLUSTER'])
        
        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df
        
        # store next_clusters dataframe
        self.nc = next_clusters(df_new)
    
    # predict() takes a list of features and a time horizon, and returns 
    # the predicted value after all actions are taken in order
    def predict(self, 
                features, # list: list OR array of features 
                actions): # list: list of actions
        
        # predict initial cluster
        s = int(self.m.predict([features]))
        
        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df,
                                        self.R_df,
                                        s,
                                        actions)
        return v
    
    # predict_forward() takes an ID & actions, and returns the predicted value
    # for this ID after all actions are taken in order
    def predict_forward(self,
                        ID,
                        actions):
        
        # cluster of this last point
        s = self.df_trained[self.df_trained['ID']==ID].iloc[-1, -2]
        
        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df,
                                        self.R_df,
                                        s,
                                        actions)
        return v
    
    
    # testing_error() takes a df_test, then computes and returns the testing 
    # error on this trained model 
    def testing_error(self, 
                      df_test,
                      relative=False,
                      h=-1):
        
        error = testing_value_error(df_test, 
                            self.df_trained, 
                            self.m, 
                            self.pfeatures,
                            relative=relative,
                            h=h)
        
        return error
    
    
    # solve_MDP() takes the trained model as well as parameters for gamma, 
    # epsilon, whether the problem is a minimization or maximization one, 
    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not 
    # represent enough percentage of all the potential next_states 
    # and returns the the value and policy. 
    def solve_MDP(self,
                  min_action_obs = 7, # int: least number of actions that must be seen
                  min_action_purity = 0.3, # float: percentage purity above which is acceptable
                  prob='max', 
                  gamma=0.9, 
                  epsilon=10**(-10),
                  p=True):
        
        # adding two clusters: one for reward sink, one for incorrectness sink
        # reward sink is R[s-2], incorrectness sink is R[s-1]
        
        P_df = self.P_df.copy()
        P_df['count'] = self.nc['count']
        P_df['purity'] = self.nc['purity']
        P_df = P_df.reset_index()
        
        # record parameters of transition dataframe
        a = P_df['ACTION'].nunique()
        s = P_df['CLUSTER'].nunique()
        n = P_df['NEXT_CLUSTER'].nunique()
        
        # Take out rows where actions or purity below threshold
        P_opt = P_df.loc[(P_df['count']>min_action_obs)&(P_df['purity']>min_action_purity)]
        
        
        # FIX to make sure there are no indexing errors - not big enough matrix defined?
        P = np.zeros((a, s+1, s+1))
        
        
        # model tranistions
        for index, row in P_opt.iterrows():
            x, y, z = row['ACTION'], row['CLUSTER'], row['NEXT_CLUSTER']
            P[x, y, z] = 1 
                
        # reinsert transition for cluster/action pairs taken out by threshold
        # ALSO INCLUDE NOT SEEN??
        excl = P_df.loc[(P_df['count']<=min_action_obs)|(P_df['purity']<min_action_purity)]
        for index, row in excl.iterrows():
            c, u = row['CLUSTER'], row['ACTION']
            P[u, c, s-1] = 1
        
        # replacing correct sink node transitions
        nan = P_df.loc[P_df['count'].isnull()]
        for index, row in nan.iterrows():
            c, u, t = row['CLUSTER'], row['ACTION'], row['NEXT_CLUSTER']
            P[u, c, t] = 1
        
        # append high negative reward for incorrect / impure transitions
        R = []
        for i in range(a):
            R.append(np.append(np.array(self.R_df),-100))
        R = np.array(R)
        
        v, pi = SolveMDP(P, R, gamma, epsilon, p, prob)
        
        # store values and policies and matrices
        self.v = v
        self.pi = pi
        self.P = P
        self.R = R
        
        return v, pi
    
    # opt_model_trajectory() takes a start state, a transition function, 
    # indices of features to be considered, a transition function, and an int
    # for number of points to be plotted. Plots and returns the transitions
    def opt_model_trajectory(self,
                             x, # start state as tuple or array
                             f, # transition function of the form f(x, u) = x'
                             f1=0, # index of feature 1 to be plotted
                             f2=1, # index of feature 2 to be plotted
                             n=30): # points to be plotted
    
        xs, ys = model_trajectory(self, f, x, f1, f2, n)
        return xs, ys
    