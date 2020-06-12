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

import sys
sys.path.append('../Algorithm/')

from MDPtools import *
from model import MDP_model
from HIV_functions import *
from clustering import *
from testing import *
#################################################################

# Set Parameters
N = 100
T = 200
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 30000
random_state = 0
pfeatures = 6
h = -1
max_k = 2
cv = 5
th = 0
classification = 'DecisionTreeClassifier' 
thresh = 2000 # threshold for dist when deciding risk

#################################################################
# Create or Load Data
#df = createSamples(N, T, 1, 'dist', thresh)
#print(df.groupby(['RISK'])['ACTION'].count())
#df.to_csv('HIV_synthetic_data.csv')
df = pd.read_csv('HIV_synthetic_data.csv')

#################################################################
# Run Algorithm
m = MDP_model()
m.fit(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
    pfeatures, # int: number of features
    h, # int: time horizon (# of actions we want to optimize)
    max_k, # int: number of iterations
    distance_threshold, # clustering diameter for Agglomerative clustering
    cv, # number for cross validation
    th, # splitting threshold
    classification, # classification method
    clustering,# clustering method from Agglomerative, KMeans, and Birch
    n_clusters, # number of clusters for KMeans
    random_state,
    plot=True)

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
