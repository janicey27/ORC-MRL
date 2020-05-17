# -*- coding: utf-8 -*-
"""
Created on Sun May 17 00:29:59 2020

@author: 
    
    
This is same as main function, but used not for one test set but for cross-validation
"""


# -*- coding: utf-8 -*-

"""

Main MDP clustering algorithm running.

"""

#################################################################
# Set Working Directory

#import os
#os.chdir("C:/Users/omars/Desktop/Georgia/opioids/iterativeMDP/") # To change
from MDPtools import Generate_random_MDP, sample_MDP_with_features_list
import numpy as np
import matplotlib.pyplot as plt
import random
#################################################################


#################################################################
# Load Libraries

from clustering import defaultNormal, UnifNormal, transformSamples, \
                        initializeClusters, splitter, split_train_test_by_id,\
                        fit_CV
from testing import *
#################################################################

random.seed(5)

#################################################################
# Set Parameters
n = 20
m = 5
reward_dep_action = False
deterministic = True
pfeatures = 2
sigma = [[0.08, 0], [0, 0.08]]
N = 400
T = 5
clustering = ''
n_clusters = 6
random_state = 0
k = n_clusters
classification = 'DecisionTreeClassifier'
n_iter = 60
th = 0 #int(0.1*N*(T-1)/n) #Threshold to stop splitting
ratio = 0.3 # portion of data to be used for testing
#################################################################


#################################################################
# Generate MDP
P, R = Generate_random_MDP(n,
                           m,
                           reward_dep_action=reward_dep_action,
                           deterministic=deterministic)

for i in range(n):
    R[i] = i%6*0.2
    
    
n_clusters = len(np.unique(R))
k = n_clusters
# Updates the correct k automatically for initial clustering based on Risk
if all(clustering != i for i in ['KMeans', 'Agglomerative', 'Birch']):
    k = len(np.unique(np.array(R)))
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

df = transformSamples(samples,
                      pfeatures)
#################################################################



#################################################################

list_training_R2,list_testing_R2 =fit_CV(df,
                                              pfeatures,
                                              k,
                                              th,
                                              clustering,
                                              classification,
                                              n_iter,
                                              n_clusters,
                                              random_state,
                                              OutputFlag = 0,
                                              n=n,
                                              cv=5)