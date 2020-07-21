#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:21:19 2020

@author: janiceyang
"""

#################################################################
# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import gym
import pickle

import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')
from MDPtools import SolveMDP
from maze_functions import policy_trajectory, policy_accuracy, fitted_Q_policy_accuracy, \
    policy, opt_model_trajectory
from model import MDP_model

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
         10: 'maze-random-10x10-plus-v0',
         11: 'maze-random-20x20-plus-v0',
         12: 'maze-random-30x30-plus-v0'}
#################################################################


#################################################################
# Import Models and Policies -- must have all the models in this folder!
Ns = [2, 3, 5]
#Ns = [2, 3, 5, 10, 20, 30, 50, 70, 90, 110, 130, 150, 170, 200]
#Ns = [10, 20]

models = [] # our trained models
policies = [] # fitted_Q policies

for n in Ns: 
    model_name = 'model_N=%i.sav'%n
    policy_name = 'fitted_Q_policy_N=%i.sav'%n
    
    m = pickle.load(open(model_name, 'rb'))
    p = pickle.load(open(policy_name, 'rb'))
    print('successful')
    
    models.append(m)
    policies.append(p)
#################################################################


#################################################################
# Plot error ('training' or 'testing') for models and policies. If error = 'testing', 
# must provide a dataframe df_test
def plot_error(models, policies, Ns, maze, error='training', df_test = None):
    model_accuracies = []
    policy_accuracies = []
    n = len(Ns)
    
    # calculate accuracies
    for i in range(n): 
        m = models[i]
        p = policies[i]
        if error == 'training':
            model_acc = policy_accuracy(m, maze, m.df)
            policy_acc = fitted_Q_policy_accuracy(p, maze, m.df)
        elif error == 'testing':
            model_acc = policy_accuracy(m, maze, df_test)
            policy_acc = fitted_Q_policy_accuracy(p, maze, df_test)
        model_accuracies.append(model_acc)
        policy_accuracies.append(policy_acc)
        print('N=', Ns[i], 'complete')
    
    # plot 
    fig1, ax1 = plt.subplots()
    ax1.plot(Ns, model_accuracies, label= "Model Accuracy")
    ax1.plot(Ns, policy_accuracies, label = "Fitted_Q Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('N training data size')
    ax1.set_ylabel('Accuracy %')
    ax1.set_title(error+' error accuracy comparison')
    ax1.legend()
    plt.show()
    return model_accuracies, policy_accuracies

        
        
    
