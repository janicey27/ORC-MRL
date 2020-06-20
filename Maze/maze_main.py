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
from maze_functions import createSamples
from testing import cluster_size, next_clusters, training_value_error, purity
#################################################################

# Set Parameters
N = 10
T_max = 50
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
random_state = 0
pfeatures = 2
actions = [0, 1, 2, 3]
h = -1
max_k = 9
cv = 5
th = 0
classification = 'DecisionTreeClassifier' 

#################################################################
# Create or Load Data

# list of maze options to choose from:
mazes = {1: 'maze-v0',
         2: 'maze-sample-3x3-v0',
         3: 'maze-random-3x3-v0',
         4: 'maze-sample-5x5-v0',
         5: 'maze-random-5x5-v0',
         6: 'maze-sample-10x10-v0',
         7: 'maze-random-10x10-v0',
         8: 'maze-sample-100x100-v0',
         9: 'maze-random-100x100-v0',
         10: 'maze-random-10x10-plus-v0', # has portals 
         11: 'maze-random-20x20-plus-v0', # has portals 
         12: 'maze-random-30x30-plus-v0'} # has portals 

df = createSamples(N, T_max, mazes[2], reseed=True)
print(df)

#################################################################
# Run Algorithm
# =============================================================================
# m = MDP_model()
# m.fit(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
#     pfeatures, # int: number of features
#     h, # int: time horizon (# of actions we want to optimize)
#     max_k, # int: number of iterations
#     distance_threshold, # clustering diameter for Agglomerative clustering
#     cv, # number for cross validation
#     th, # splitting threshold
#     classification, # classification method
#     clustering,# clustering method from Agglomerative, KMeans, and Birch
#     n_clusters, # number of clusters for KMeans
#     random_state,
#     plot=True)
# 
# =============================================================================
