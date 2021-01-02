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
from maze_functions import createSamples, opt_maze_trajectory, opt_model_trajectory
from testing import cluster_size, next_clusters, training_value_error, purity
#################################################################

# Set Parameters
N = 50
T_max = 25
max_k = 25
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
precision_thresh = 1e-14
random_state = 0
pfeatures = 2
actions = [0, 1, 2, 3]
h = -1
cv = 5
th = 0
eta = 25
gamma = 1
classification = 'DecisionTreeClassifier'
split_classifier_params = {'random_state':0, 'max_depth':2}


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


df = createSamples(N, T_max, mazes[4], 0.4, reseed=True)
#print(df)

#################################################################
# Run Algorithm

m = MDP_model()
m.fit(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
    pfeatures, # int: number of features
    h, # int: time horizon (# of actions we want to optimize)
    gamma, # discount factor
    max_k, # int: number of iterations
    distance_threshold, # clustering diameter for Agglomerative clustering
    cv, # number for cross validation
    th, # splitting threshold
    eta, # incoherence thresholds
    precision_thresh, # precision threshold for further splitting
    classification, # classification method
    split_classifier_params, # classification params
    clustering,# clustering method from Agglomerative, KMeans, and Birch
    n_clusters, # number of clusters for KMeans
    random_state,
    plot=False,
    optimize=True,
    verbose=False)


#################################################################
# Helper Scripts

# Loading csv
'''
filename = ''
df = pd.read_csv(filename)

# taking out extra ID col and changing actions back to integers
df = df.iloc[:, 1:]
df.loc[df['ACTION']=='None', 'ACTION'] = 4
df['ACTION'] = pd.to_numeric(df['ACTION'], downcast='integer')
df.loc[df['ACTION']==4, 'ACTION'] = 'None'
'''

# Running fitted_Q
'''
f, r = get_maze_transition_reward(mazes[4])
Q, p, x_df = fitted_Q(100, df, 0.98, 2, [0, 1, 2, 3], f, r, True, 'ExtraTrees')
pickle.dump(p, open('fitted_Q_policy_N=170.sav', 'wb'))
'''

# Generate a lot of csvs: 
'''
for i in range(150, 200):
    df = createSamples(N, T_max, mazes[4], 0.4, reseed=True)
    df.to_csv(f'set_{i}.csv')
'''

# Update df rewards to -0.04 for all RISK
'''
for i in range(150, 200):
    filename = f'set_{i}.csv'
    df = pd.read_csv(filename)
    
    df = df.iloc[:, 1:]
    df.loc[df['ACTION']=='None', 'ACTION'] = 4
    df['ACTION'] = pd.to_numeric(df['ACTION'], downcast='integer')
    df.loc[df['ACTION']==4, 'ACTION'] = 'None'
    
    df.loc[df['RISK']==-0.004, 'RISK'] = -0.04
    df.to_csv(filename)
'''