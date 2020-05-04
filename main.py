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
#################################################################


#################################################################
# Load Libraries

from clustering import defaultNormal, UnifNormal, transformSamples, initializeClusters, splitter
from testing import *
#################################################################


#################################################################
# Set Parameters
n = 10
m = 3
reward_dep_action = False
deterministic = True
pfeatures = 2
sigma = [[0.03, 0], [0, 0.03]]
N = 100
T = 5
clustering = ''
n_clusters = 3
random_state = 0
k = n_clusters
classification = 'DecisionTreeClassifier'
n_iter = 16
th = int(0.1*N*(T-1)/n)
#################################################################


#################################################################
# Generate MDP
P, R = Generate_random_MDP(n,
                           m,
                           reward_dep_action=reward_dep_action,
                           deterministic=deterministic)

for i in range(n-3):
    R[i] = 0
R[n-2] = 1
R[n-1] = -1

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
#################################################################


#################################################################
# Transform into DataFrame
df = transformSamples(samples,
                      pfeatures)#################################################################

df_train, df_test = split_train_test_by_id(df, 0.2, 'ID')
#################################################################
# Initialize Clusters
df = initializeClusters(df_train,
                        clustering=clustering,
                        n_clusters=n_clusters,
                        random_state=random_state)
#################################################################

#################################################################
# Run Iterative Learning Algorithm

df_new = splitter(df,
                  pfeatures,
                  k,
                  th,
                  classification,
                  n_iter)

#################################################################

print(Purity(df_new))
#plot_features(df)
#
print('training accuracy:',training_accuracy(df_new)[0])
print('training error:', training_value_error(df_new))
print('testing error:', testing_value_error(df_test, df_new, pfeatures))

#print('Training R2:', R2_value(df_new,N))
