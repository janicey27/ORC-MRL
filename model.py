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
        self.df_trained = None # dataframe after optimal training
        self.m = None # model for predicting cluster number from features
        self.P_df = None #Transition function of the learnt MDP
        self.R_df = None #Reward function of the learnt MDP
        
        
    # fit_CV() takes in parameters for prediction, and trains the model on the 
    # optimal clustering for a given horizon h
    def fit_CV(self, 
            data, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
            pfeatures, # int: number of features
            #h=5, # int: time horizon (# of actions we want to optimize)
            n_iter=70, # int: number of iterations
            distance_threshold = 0.05, # clustering diameter for Agglomerative clustering
            cv=5, # number for cross validation
            th=0, # splitting threshold
            classification = 'DecisionTreeClassifier', # classification method
            clustering='Agglomerative',# clustering method from Agglomerative, KMeans, and Birch
            n_clusters = None, # number of clusters for KMeans
            random_state = 0):
        
        df = data.copy()
            
        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        
        # run cross validation on the data to find best clusters
        list_training_error,list_testing_error, k =fit_CV(self.df,
                                              self.pfeatures,
                                              th,
                                              clustering,
                                              distance_threshold,
                                              classification,
                                              n_iter,
                                              n_clusters,
                                              random_state,
                                              #h = h,
                                              OutputFlag = 0,
                                              cv=cv)
        
        # find the best cluster
        cv_testing_error = np.mean(np.array(list_testing_error),axis=0)
        it = np.argmin(cv_testing_error)
        print('minimum iterations:', it+1) # this is iterations, but should be cluster
        print('best clusters:', k+it+1)
        
        # save total error and error corresponding to chosen model
        self.CV_error = cv_testing_error[it]
        self.CV_error_all = cv_testing_error
        self.opt_nc = k+it+1
        
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
                                          opt_nc = self.opt_nc,
                                          classification=classification,
                                          it = n_iter,
                                          #h=h,
                                          OutputFlag = 0)
        
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
    
    # take an ID & actions, and predict what will happen from there
    def predict_forward(self,
                        ID,
                        actions):
        pass