# -*- coding: utf-8 -*-

"""

Main MDP clustering algorithm running.

"""

#################################################################
# Set Working Directory

#import os
#os.chdir("C:/Users/omars/Desktop/Georgia/opioids/iterativeMDP/") # To change

import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')

from model import MDP_model
from MDPtools import Generate_random_MDP, sample_MDP_with_features_list
#################################################################


#################################################################
# Load Libraries

from testing import purity, plot_features, next_clusters
from grid_functions import UnifNormal, transformSamples
# find a way to import grid_functions into the actual model which calls on splitter.......
#################################################################

random.seed(5)

#################################################################
# Set Parameters
n = 15
m = 3
reward_dep_action = False
deterministic = True
pfeatures = 2
sigma = [[0.01, 0], [0, 0.01]]
N = 250
T = 5
clustering = 'Agglomerative'
n_clusters = None # for KMeans
random_state = 0
classification = 'DecisionTreeClassifier'
split_classifier_params = {'random_state':0}
max_k = 15
th = 0 #int(0.1*N*(T-1)/n) #Threshold to stop splitting
ratio = 0.3 # portion of data to be used for testing
cv = 5
distance_threshold = 0.01
h = -1 # time horizon we want to optimize
#################################################################


#################################################################
# Generate MDP
P, R = Generate_random_MDP(n,
                           m,
                           reward_dep_action=reward_dep_action,
                           deterministic=deterministic)

for i in range(n):
    R[i] = i%6*0.2
    
    
#n_clusters = len(np.unique(R))
#################################################################


#################################################################
# Generate Samples from Normal Distribution

#normal_distributions = defaultNormal(n,
#                                     pfeatures,
#                                     sigma)

normal_distributions = UnifNormal(n,
                                     pfeatures,
                                     sigma)


samples = sample_MDP_with_features_list(P,
                                        R,
                                        normal_distributions,
                                        N,
                                        T)
#################################################################


#################################################################
# Transform into Training and Testing DataFrames
df = transformSamples(samples,
                      pfeatures)

m = MDP_model()
m.fit_CV(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
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
    plot=True)
