#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:03:52 2020

@author: janiceyang
"""

#################################################################
# Set Working Directory
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle

import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')


#from MDPtools import *
from model import MDP_model
from HIV_functions import createSamples
#from clustering import *
from testing import cluster_size, next_clusters, training_value_error
#################################################################

# Set Parameters
N = 100
T = 200
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
random_state = 0
pfeatures = 6
h = -1
max_k = 500
cv = 5
th = 0
classification = 'DecisionTreeClassifier' 
split_classifier_params = {'random_state':0, 'max_depth':2}
thresh = 2000 # threshold for dist when deciding risk

#################################################################
# Create or Load Data
#df = createSamples(N, T, 1, 'c_r', None)
#print(df.groupby(['RISK'])['ACTION'].count())
#df.to_csv('HIV_synthetic_data_large.csv')
#df = pd.read_csv('HIV_synthetic_data_large.csv')
df = pd.read_csv('df_HIV.csv')
print('loaded data')

#################################################################
# Run Algorithm
m = MDP_model()
m.fit(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'ACTION', 'RISK']
    pfeatures, # int: number of features
    h, # int: time horizon (# of actions we want to optimize)
    max_k, # int: number of iterations
    distance_threshold, # clustering diameter for Agglomerative clustering
    cv, # number for cross validation
    th, # splitting threshold
    classification, # classification method
    split_classifier_params,
    clustering,# clustering method from Agglomerative, KMeans, and Birch
    n_clusters, # number of clusters for KMeans
    random_state,
    plot=True,
    optimize=True,
    OutputFlag=0)

pickle.dump(m, open('m1_fit_opt.sav', 'wb'))

#cs = cluster_size(m.df_trained)
#nc = next_clusters(m.df_trained)

#plt.plot([training_value_error(m.df_trained, True, h) for h in range(0, 15)])
#print(cs)
#print(nc)

# prints the state and cost path of a patient taking h steps, starting at 
# unhealthy steady state, and only taking action u
def path(h, u):
    x_init = (163573, 5, 11945, 46, 63919, 24)
    s = int(m.m.predict(np.array(x_init).reshape(1, -1)))
    print(s, m.R_df[s])
    for i in range(h):
        s = m.P_df.loc[s, u].values[0]
        print(s, m.R_df[s])

# =============================================================================
# P, R = Generate_random_MDP(5, 3, reduced=False, reward_dep_action=True) # (states, actions)
# print(P, R)
# v, pi = SolveMDP(P, R)
# # NEED TO FIX SolveMDP when m and n not given
# =============================================================================


# =============================================================================
# 
# # Testing Distance Thresholds
# dt = [100000]
# clusters = []
# for distance_threshold in dt:
#     df_init = initializeClusters(df,
#                                 clustering=clustering,
#                                 n_clusters=n_clusters,
#                                 distance_threshold = distance_threshold,
#                                 random_state=random_state)
#     
#     k = df_init['CLUSTER'].nunique()
#     print(df_init.groupby(['CLUSTER'])['RISK'].count())
#     print('clusters', k)
#     print('mean points per cluster', df_init.groupby(['CLUSTER'])['RISK'].count().mean())
#     clusters.append(cluster_size(df_init))
# =============================================================================


#P, R = get_MDP(df_init)
