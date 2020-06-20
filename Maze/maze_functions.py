#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:57:14 2020

@author: janiceyang
"""
#################################################################
# Load Libraries
import sys
import numpy as np
import pandas as pd
import math
import random
import gym

# If Spyder keeps re-calling gym_maze module, run the following code
'''
# deleting the registry that spyder has already 
for env in gym.envs.registry.env_specs.copy():
    #print(env)
    if 'maze' in env:
        print('Remove {} from registry'.format(env))
        del gym.envs.registry.env_specs[env]
'''
import gym_maze
#################################################################

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


# createSamples() N, the number of IDs, T_max, the maximum timestep if win-
# state is not reached, and maze (str of maze-name from above dict). If 
# reseed = True, selects a random location in the next cell, otherwise reseed=
# False makes robot travel to the next cell with the same offset
# returns a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
def createSamples(N, T_max, maze, reseed=False):
    
    # initialize environment
    env = gym.make(maze)
    transitions = []
    
    for i in range(N):
        
        # initialize variables 
        offset = np.array((random.random(), random.random()))
        obs = env.reset()
        
        # reward for SMALL 3x3 MAZE CHANGE LATER
        reward = -0.1/(env.maze_size[0]*env.maze_size[1])
        x = obs + offset
        
        for t in range(T_max):
            action = env.action_space.sample()
            
            transitions.append([i, t, x, action, reward])
            
            new_obs, reward, done, info = env.step(action)
            
            # if reseed, create new offset
            if reseed:
                offset = np.array((random.random(), random.random()))
                
            # if end state reached, append one last no action no reward
            if done:
                transitions.append([i, t+1, new_obs+offset, 'None', reward])
                break
            x = new_obs + offset
    
    df = pd.DataFrame(transitions, columns=['ID', 'TIME', 'x', 'ACTION', 'RISK'])
                      
    features = df['x'].apply(pd.Series)
    features = features.rename(columns = lambda x : 'FEATURE_' + str(x))
    
    df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
    
    return df_new
                      
    

# Initialize the "maze" environment
# =============================================================================
# env = gym.make("maze-random-10x10-plus-v0")
# obs = []
# 
# # first point for ID
# observation = env.reset()
# offset = np.array((random.random(), random.random()))
# #env.state = offset
# #print(env.state)
# obs.append(observation+offset)
# for _ in range(1000):
#     
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     print(observation, reward, done, info)
#     obs.append(observation+offset)
#       
#     if done:
#       observation = env.reset()
# env.close()
# =============================================================================
