
#################################################################
# Set Working Directory
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle


#from MDPtools import *
from model import MDP_model
from maze_functions import createSamples, opt_maze_trajectory, opt_model_trajectory, fitted_Q, \
    get_maze_transition_reward, policy_accuracy, fitted_Q_policy_accuracy
from testing import cluster_size, next_clusters, training_value_error, purity


# Set Parameters
#N = 170
T_max = 25
max_k = 25
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
random_state = 0
pfeatures = 2
actions = [0, 1, 2, 3]
h = -1
cv = 5
th = 0
classification = 'DecisionTreeClassifier'
#classification = 'RandomForestClassifier'
split_classifier_params = {'random_state':0, 'max_depth':2}

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

# creating the necessary f and r functions
P = pickle.load(open('5x5_P.sav', 'rb'))
R = pickle.load(open('5x5_Reward.sav', 'rb'))

l = int((R.size/4-1)**0.5)

# importing necessary functions
def f(x, u, plot=True):
    #print('x', x, 'u', u)
    
    # if sink, return None again
    if plot:
        if x[0] == None:
            return (None, None)
    
    # if no action, or an action '4' to simulate no action: 
    if u == 'None' or u == 4:
        if plot:
            return (None, None)
        else:
            return (0, -l)
    
    # first define the cluster of the maze based on position
    x_orig = (int(x[0]), int(-x[1]))
    offset = np.array((random.random(), -random.random()))

    
    c = int(x_orig[0] + x_orig[1]*l)
    c_new = P[u, c].argmax()
    
    
    # if maze at sink, return None
    if c_new == R.size/4-1:
        if plot:
            return (None, None)
        else:
            return (0, -l)
    else:
        x_new = (c_new%l, c_new//l)
        x_new = (x_new[0], -x_new[1])
    return x_new + offset
    
    
def r(x):
    # if sink, return 0 reward
    if x[0] == None:
        return R[0][-1]
    else:
        x_orig = (int(x[0]), int(-x[1]))
        c = int(x_orig[0] + x_orig[1]*l)
        #print(c)
        return R[0][c]

# Set Ns we want to investigate
Ns = [5, 10, 20, 30, 35, 40, 45, 50, 70, 90, 110, 130, 150, 170, 200]

# Training models and fitted Q! 
i = 0 
while True:
    # read file and deal with 'None' values etc.
    df_full = pd.read_csv('set_{}.csv'.format(i))
    df_full = df_full.iloc[:, 1:]
    df_full.loc[df_full['ACTION']=='None', 'ACTION'] = 4
    df_full['ACTION'] = pd.to_numeric(df_full['ACTION'], downcast='integer')
    df_full.loc[df_full['ACTION']==4, 'ACTION'] = 'None'
    
    # creating new dataset and saving model and policies
    for n in Ns:
        df = df_full.loc[df_full['ID']<n]
        
        m = MDP_model()
        m.fit_CV(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
            pfeatures, # int: number of features
            h, # int: time horizon (# of actions we want to optimize)
            max_k, # int: number of iterations
            distance_threshold, # clustering diameter for Agglomerative clustering
            cv, # number for cross validation
            th, # splitting threshold
            classification, # classification method
            split_classifier_params, # classification params
            clustering,# clustering method from Agglomerative, KMeans, and Birch
            n_clusters, # number of clusters for KMeans
            random_state,
            plot=False)
        pickle.dump(m, open('round_{}_model_N={}.sav'.format(i, n), 'wb'))
        
        Q, p, x_df = fitted_Q(30, df, 0.98, 2, [0, 1, 2, 3], f, r, True, 'ExtraTrees')
        pickle.dump(p, open('round_{}_fqpolicy_N={}.sav'.format(i, n), 'wb'))
        print('N=', n, ' completed')
    
    print('Round', i, 'completed')
    i += 1



