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
from datetime import timedelta

from clustering import *
from testing import *
#################################################################

class MDP_model:
    def __init__(self):
        self.df = None # original dataframe from data
        self.pfeatures = None # number of features
        self.CV_error = None # error at minimum point of CV
        self.CV_error_all = None # errors of different clusters after CV
        self.training_error = None # training error after very last split
        self.opt_k = None # number of clusters in optimal clustering
        self.df_trained = None # dataframe after optimal training
        self.m = None # model for predicting cluster number from features
        self.P_df = None #Transition function of the learnt MDP
        self.R_df = None #Reward function of the learnt MDP
        
        
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
        
        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = self.opt_k,
                                          classification=classification,
                                          h=h,
                                          OutputFlag = 0,
                                          plot = plot)
        
        # store final training error
        try:
            # if splitter doesn't need to split at all, then no training error
            self.training_error = training_error.iloc[-1]
        except:
            pass
        
        
        # storing trained dataset and predict_cluster function
        self.df_trained = df_new
        self.m = predict_cluster(self.df_trained, self.pfeatures)
        
        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df
        
    
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
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0,
            plot = False):
    
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
        
        df_new,training_error,testing_error = splitter(df_init,
                                          pfeatures=self.pfeatures,
                                          th=th,
                                          df_test = None,
                                          testing = False,
                                          max_k = max_k,
                                          classification=classification,
                                          h=h,
                                          OutputFlag = 0,
                                          plot = plot)
        
        # store final training error
        self.training_error = training_error.iloc[-1]
        
        # storing trained dataset and predict_cluster function
        self.df_trained = df_new
        self.m = predict_cluster(self.df_trained, self.pfeatures)
        
        # store P_df and R_df values
        P_df,R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df
    
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