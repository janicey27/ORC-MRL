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

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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

# reward_c() does the same thing as reward, except takes an x tuple that is 
# in cartesian (x,y) coordinates
def reward_c(x):
    if (x[0]**2+x[1]**2)**0.5 <= 1:
        if x[0] > 0 and x[1] > 0:
            return 1
        else:
            return 0
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
    

# fitted_Q() trains K functions Q1 to QK that determine the optimal strategy
# x_df is a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
# for each one-step transition returns the last iteration QK, and a function \
# policy that takes a state and outputs the optimal action
def fitted_Q(K, # number of iterations
             x_df, # dataframe 
             gamma, # decay factor
             pfeatures, # number of features in dataframe
             actions, # list of action possibilities
             r_max, # r_max of this toy example
             take_max = True, # True if max_cost is good, otherwise false
             regression = 'Linear Regression' # str: type of regression
             ):
    
    x_df = x_df.copy(deep=True)
    # initialize storage and actions
    #Qs = []
    
    # create the first Q1 function
    class Q:
        def predict(self, array): 
            x = array[0][:pfeatures]
            #print('x', x)
            return reward_c(x)
    Q_new = Q()
    #Qs.append(Q_new)
    
    # calculate the x_t2 next step for each x
    x_df['x_t2'] = x_df.apply(lambda x: list(f(tuple(x[2:2+pfeatures]), x.ACTION, \
                                          r_max)), axis=1)
    for i in range(len(actions)):
        x_df['a%s'%i] = x_df.apply(lambda x: x.x_t2+[actions[i]], axis=1)
    action_names = ['a%s'%i for i in range(len(actions))]
    
    print(x_df, flush=True)
    # create X using x_t and u
    # select (x_t, u) pair as training
    X = x_df.iloc[:, 2:3+pfeatures]
    print('New training features', flush=True)
    print(X, flush=True)
    
    
    bar = tqdm(range(K))
    #bar = range(K)
    #creating new Qk functions
    for i in bar:
        # create y using Qk-1 and x_t2
        # non-DP
        if take_max: 
            y = x_df.apply(lambda x: x.RISK + gamma*max([Q_new.predict([f]) \
                                for f in [x[a] for a in action_names]]), axis=1)
        else:
            y = x_df.apply(lambda x: x.RISK + gamma*min([Q_new.predict([f]) \
                                for f in [x[a] for a in action_names]]), axis=1)
        
        print(y, flush=True)
        '''                               
        # initialize dp
        memo = {}
        mu = 0
        y = []                        
        for index, row in x_df.iterrows():
            qs = []
            for f in [row['a0'], row['a1'], row['a2'], row['a3']]:
                if f in memo:
                    qs.append(memo[f])
                    #print('memo used')
                    mu += 1
                else:
                    q = Q_new.predict([f])
                    memo[f] = q
                    qs.append(memo[f])
            y.append(row['c'] + gamma*min(qs))
        '''
        
        y = np.array(y)
        #print(y)
        
        # train the actual Regression function as Qk
        #regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
        if regression ==  'LinearRegression':
            regr = LinearRegression().fit(X, y)
        if regression == 'RandomForest':
            regr = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
        if regression == 'ExtraTrees':
            regr = ExtraTreesRegressor(n_estimators=50).fit(X,y.ravel())
        #Qs.append(regr)
        Q_new = regr
        #print('memo size', len(memo), 'used', mu, flush=True)
        
    
    #QK = Qs[-1]
    QK = Q_new
    
    p = policy(actions, take_max)
    p.fit(QK)
        
    return QK, p#, Qs


class policy:
    def __init__(self, actions, take_max):
        self.QK = None
        self.actions = actions
        self.take_max = take_max
        
    def fit(self, 
            QK): # model, the latest fitted_Q 
        self.QK = QK
    
    # pred() takes a state x and predicts the optimal action
    def get_action(self,
             x): 
        if self.take_max:
            i = np.argmax([self.QK.predict([x + [u]]) \
                                        for u in self.actions])
        else:
            i = np.argmin([self.QK.predict([x + [u]]) \
                                            for u in self.actions])
        return self.actions[i]