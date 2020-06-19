#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:37:12 2020

@author: janiceyang
"""

#################################################################
# Load Libraries
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
#################################################################

# f() simulates the transitions for actions 0 and 1. It takes a state x which
# is a (r, theta) pair, then performs transition for actions inwards by 1 unit
# (action 0), or outwards by 1 unit (action 1).
# returns the new x tuple
def f(x, a, r_max):

    # if x in the reward sink already, no change
    if x[0] <= 0:
        return x
    # move those who just got to reward to the sink (r <= 0)
    elif x[0] <= 1: 
        return x[0]-1, x[1]
    else:
        if a == 0:
            return x[0]-1, x[1]
        elif a == 1:
            if x[0]+1 > r_max:
                return x
            else:
                return x[0]+1, x[1]
        else:
            print('Action not seen')
            return 
        
# reward() takes a state x and returns the corresponding reward of reaching
# this state. Reward is only 1 when 0 < r <= 1
def reward(x):
    if x[0] > 0 and x[0] <= 1:
        return 1
    else: 
        return 0
    
    
# createSamples() generates dataset that starts at random position in the first 
# quadrant grid (r, theta), for N trajectories of T steps each, with starting values 
# of r where 0 < r < r_max. returns a DataFrame of samples in the form: ['ID',
# 'TIME', 'x', 'ACTION', 'RISK']
def createSamples(N, T, r_max):
    
    transitions = []
    for i in range(N):
        # can start anywhere, even in reward box!
        x = (random.uniform(0, r_max), random.uniform(0,1)*np.pi/2)
        x_c = (x[0]*np.cos(x[1]), x[0]*np.sin(x[1]))
        for t in range(T):
            if random.random() < 0.5:
                a = 0
            else:
                a = 1
            x_t = f(x, a, r_max)
            x_tc = (x_t[0]*np.cos(x_t[1]), x_t[0]*np.sin(x_t[1]))
            c = reward(x)
            if x[0] < 0:
                ogc = 0
            else:
                ogc = int(x[0])+1
            transitions.append([i, t, x_c, a, c, ogc])
            x = x_t
            x_c = x_tc
    
    df = pd.DataFrame(transitions, columns=['ID', 'TIME', 'x', 'ACTION', 'RISK',\
                                            'OG_CLUSTER'])
    #print(df)
    features = df['x'].apply(pd.Series)
    features = features.rename(columns = lambda x : 'FEATURE_' + str(x))
    
    df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
    
    return df_new
    
    