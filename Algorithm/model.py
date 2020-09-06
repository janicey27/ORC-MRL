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
from scipy.stats import binom
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
        self.m = None # model for predicting cluster number from features #CHANGE NAME
        self.clus_pred_accuracy = None # accuracy score of the cluster prediction function
        self.P_df = None # Transition function of the learnt MDP, includes sink node if end state exists
        self.R_df = None # Reward function of the learnt MDP, includes sink node of reward 0 if end state exists
        self.nc = None # dataframe similar to P_df, but also includes 'count' and 'purity' cols
        self.v = None # value after MDP solved
        self.pi = None # policy after MDP solved
        self.P = None # P_df but in matrix form of P[a, s, s'], with alterations
                        # where transitions that do not pass the action and purity thresholds
                        # now lead to a new cluster with high negative reward
        self.R = None # R_df but in matrix form of R[a, s]
        
        
    # fit_CV() takes in parameters for prediction, and trains the model on the 
    # optimal clustering for a given horizon h (# of actions), using cross
    # validation.
    def fit_CV(self, 
            data, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
            pfeatures, # int: number of features
            h=5, # int: time horizon (# of actions we want to optimize)
            gamma=1, # discount value
            max_k=70, # int: max number of clusters
            distance_threshold = 0.05, # clustering diameter for Agglomerative clustering
            cv=5, # number for cross validation
            th=0, # splitting threshold
            classification = 'DecisionTreeClassifier', # classification method
            split_classifier_params = {'random_state': 0},
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0,
            plot = False,
            OutputFlag = 0):
        
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
                                                  gamma = gamma, 
                                                  OutputFlag = OutputFlag,
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
        
        df_new,training_error,testing_error, best_df = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = self.opt_k,
                                          classification=classification,
                                          split_classifier_params = split_classifier_params,
                                          h=h,
                                          gamma=gamma,
                                          OutputFlag = OutputFlag,
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
            gamma=1, # discount value
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
            optimize = True,
            OutputFlag = 0):
    
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
        
        df_new,training_error,testing_error, best_df = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = max_k,
                                          classification=classification,
                                          split_classifier_params = split_classifier_params,
                                          h=h,
                                          gamma=gamma,
                                          OutputFlag = OutputFlag,
                                          plot = plot)
        
        # store all training errors
        self.training_error = training_error
        
        # # if optimize, find best cluster and resplit
        # if optimize: 
        #     k = self.training_error['Clusters'].iloc[self.training_error['Error'].idxmin()]
        #     for i in range(k):
        #         # if clustering is less than k but error within 10^-14, take this
        #         if abs(self.training_error['Error'].min() - \
        #                self.training_error.loc[self.training_error['Clusters']==i]['Error'].min()) < 1e-14:
        #             k = i
        #             break
        #     self.opt_k = k
        #     df_new,training_error,testing_error = splitter(df_init,
        #                                   pfeatures=self.pfeatures,
        #                                   th=th,
        #                                   df_test = None,
        #                                   testing = False,
        #                                   max_k = self.opt_k,
        #                                   classification=classification,
        #                                   split_classifier_params = split_classifier_params,
        #                                   h=h,
        #                                   gamma=gamma,
        #                                   OutputFlag = OutputFlag,
        #                                   plot = plot)
        
        # storing trained dataset and predict_cluster function
        if optimize:
            self.df_trained = best_df
            k = self.training_error['Clusters'].iloc[self.training_error['Error'].idxmin()]
            self.opt_k = k
        else:
            self.df_trained = df_new
            
        self.m = predict_cluster(self.df_trained, self.pfeatures)
        pred = self.m.predict(self.df_trained.iloc[:, 2:2+self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, self.df_trained['CLUSTER'])
        
        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df
        
        # store next_clusters dataframe
        self.nc = next_clusters(self.df_trained)
    
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
                  alpha = 0.2, # statistical alpha threshold
                  beta = 0.5, # statistical beta threshold
                  min_action_obs = 5, # int: least number of actions that must be seen
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
        actions = P_df['ACTION'].unique()
        
        # convert ACTION, CLUSTER, and NEXT_CLUSTER to integers
        
        
        #print(P_df)
        # Take out rows that don't pass statistical alpha test
        P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
                                      beta))<=alpha]
        
        # for old version of self.nc:
        #P_alph = P_df.loc[(1-binom.cdf(P_df['count'], P_df['count']/P_df['purity'],\
                                      #0.5))<=alpha]
        
        # Take out rows where actions or purity below threshold
        P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]
        
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(P_df.groupby('CLUSTER')['ACTION'].count()<a)[0]
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(actions, P_df.loc[P_df['CLUSTER']==c]['ACTION'].unique())
            for u in not_present:
                missing_pairs.append((c, u))
        
        #print(P_opt)
        
        # FIX to make sure there are no indexing errors - not big enough matrix defined?
        P = np.zeros((a, s+1, s+1))
        
        
        # model transitions
        for row in P_thresh.itertuples():
            x, y, z = row[2], row[1], row[3] #ACTION, CLUSTER, NEXT_CLUSTER
            P[x, y, z] = 1 
                
        # reinsert transition for cluster/action pairs taken out by alpha test
        excl_alph = P_df.loc[(1-binom.cdf(P_df['purity']*P_df['count'], P_df['count'],\
                                      beta))>alpha]
        for row in excl_alph.itertuples():
            c, u = row[1], row[2] #CLUSTER, ACTION
            P[u, c, -1] = 1
            
        # reinsert transition for cluster/action pairs taken out by threshold
        # ALSO INCLUDE NOT SEEN??
        excl = P_df.loc[(P_df['count']<=min_action_obs)|(P_df['purity']<=min_action_purity)]
        for row in excl.itertuples():
            c, u = row[1], row[2] #CLUSTER, ACTION
            P[u, c, -1] = 1
            
        # reinsert transition for missing cluster-action pairs
        for pair in missing_pairs:
            c, u = pair
            P[u, c, -1] = 1
                
        
        # replacing correct sink node transitions
        nan = P_df.loc[P_df['count'].isnull()]
        for row in nan.itertuples():
            c, u, t = row[1], row[2], row[3] #CLUSTER, ACTION, NEXT_CLUSTER
            P[u, c, t] = 1
        
        # punishment node to itself:
        for u in range(a):
            P[u, -1, -1] = 1
        
        # append high negative reward for incorrect / impure transitions
        R = []
        for i in range(a):
            if prob == 'max':
                # instead of -100, take T-max * max reward * 10 ## store T_max and r_max from fit
                R.append(np.append(np.array(self.R_df),-100))
            else:
                R.append(np.append(np.array(self.R_df),100))
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
    
    
    # update_nc() allows self.nc to be updated for models that were saved 
    # before the 'COUNT' issue in next_clusters was resolved
    def update_nc(self):
        self.nc = next_clusters(self.df_trained)
        return
    