# -*- coding: utf-8 -*-

"""

This file is the main file to run the MDP clustering algorithm



 on data for the MIT-Lahey Opioids project.



Created on Sun Mar  1 18:51:20 2020



@author: omars

"""



#################################################################

# Set Working Directory

#import os

#os.chdir("C:/Users/omars/Desktop/Georgia/opioids/iterativeMDP/") # To change

from MDPtools import Generate_random_MDP, sample_MDP_with_features_list

import numpy as np

#################################################################





#################################################################

# Load Libraries

from clustering import defaultNormal, UnifNormal, transformSamples, initializeClusters, splitter, Purity, plot_features

#################################################################





#################################################################

# Set Parameters

n = 50

m = 3

reward_dep_action = False

deterministic = True

pfeatures = 2

sigma = [[0.01, 0], [0, 0.01]]

N = 100000

T = 2

clustering = ''

n_clusters = 3

random_state = 0

k = n_clusters

classification = 'DecisionTreeClassifier'

n_iter = 60


th = 20

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

                      pfeatures)
df0 = df.copy()
#################################################################





#################################################################

# Initialize Clusters

df = initializeClusters(df,

                        T,

                        clustering=clustering,

                        n_clusters=n_clusters,

                        random_state=random_state)

#################################################################





#################################################################

# Run Iterative Learning Algorithm

df_new = splitter(df,

                  T,

                  pfeatures,

                  k,

                  th,

                  classification,

                  n_iter)

#################################################################
print(Purity(df_new))
plot_features(df)
