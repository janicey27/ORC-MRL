#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:20:32 2020

@author: janiceyang
"""

#################################################################
# Set Working Directory
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')

#from MDPtools import *
from model import MDP_model
from toy_functions import createSamples, fitted_Q
from testing import cluster_size, next_clusters, training_value_error, purity
#################################################################

# Set Parameters
N = 50
T = 100
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
random_state = 0
pfeatures = 2
actions = [0, 1]
h = -1
max_k = 11
cv = 5
th = 0
classification = 'DecisionTreeClassifier' 
thresh = 2000 # threshold for dist when deciding risk
r_max = 10 # max number of bands, actual number of states is r_max + 1 from sink

#################################################################
# Create or Load Data
df = createSamples(N, T, r_max)
print(df)

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
    plot=True,
    optimize=False)
#################################################################
# Run Fitted_Q

# =============================================================================
# Q, p = fitted_Q(50, # number of iterations
#              df, # dataframe 
#              0.98, # decay factor
#              pfeatures, # number of features in dataframe
#              actions, # list of action possibilities
#              r_max = r_max, # r_max of this toy example
#              take_max = True, # True if max_cost is good, otherwise false
#              regression = 'Linear Regression') # str: type of regression
# 
# x = [random.uniform(0, 10), random.uniform(0, 10)]
# p.get_action(x)
# =============================================================================
