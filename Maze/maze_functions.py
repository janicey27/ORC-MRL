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
import matplotlib.pyplot as plt
import math
import random
import gym

import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')
from MDPtools import *

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
# state is not reached, and maze (str of maze-name from above dict). Generates 
# the samples based on a ratio r of randomness, and (1-r) actions taken according
# to the optimal policy of the maze. If 
# reseed = True, selects a random location in the next cell, otherwise reseed=
# False makes robot travel to the next cell with the same offset
# returns a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
def createSamples(N, T_max, maze, r, reseed=False):
    # solving for optimal policy
    P, R = get_maze_MDP(maze)
    v, pi = SolveMDP(P, R, 0.98, 1e-10, False, 'max')
    
    # initialize environment
    env = gym.make(maze)
    transitions = []
    l = env.maze_size[0]
    
    for i in range(N):
        
        # initialize variables 
        offset = np.array((random.random(), random.random()))
        obs = env.reset()
        
        # initialize first reward
        reward = -0.1/(l*l)
        x = obs + offset
        ogc = int(obs[0] + obs[1]*l)
        
        for t in range(T_max):
            # take random step or not
            if random.random() <= r:
                action = env.action_space.sample()
            else:
                action = int(pi[ogc])
            
            transitions.append([i, t, x, action, reward, ogc])
            
            new_obs, reward, done, info = env.step(action)
            ogc = int(new_obs[0] + new_obs[1]*l)
            
            # if reseed, create new offset
            if reseed:
                offset = np.array((random.random(), random.random()))
                
            # if end state reached, append one last no action no reward
            if done:
                transitions.append([i, t+1, new_obs+offset, 'None', reward, ogc])
                break
            x = new_obs + offset
            
    
    df = pd.DataFrame(transitions, columns=['ID', 'TIME', 'x', 'ACTION', 'RISK', 'OG_CLUSTER'])
                      
    features = df['x'].apply(pd.Series)
    features = features.rename(columns = lambda x : 'FEATURE_' + str(x))
    
    df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
    df_new['FEATURE_1'] = -df_new['FEATURE_1']
    
    return df_new
             

# trajectory() takes a trained model, the maze used to train this model, and 
# plots the path of the optimal solution through the maze. returns the path
def trajectory(m, maze):
    if m.v is None:
        m.solve_MDP()
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    reward = -0.1/(l*l)
    
    xs = [obs[0]]
    ys = [-obs[1]]
    done = False
    offset = np.array([0.5, -0.5])
    point = np.array((obs[0], -obs[1])) + offset

    while not done:
        # find current state and action
        s = m.m.predict(point.reshape(1, -1))
        #print(s)
        a = int(m.pi[s])
        #print(a)
        
        obs, reward, done, info = env.step(a)
        point = np.array((obs[0], -obs[1])) + offset
        #print(done)
        xs.append(obs[0])
        ys.append(-obs[1])

    xs = np.array(xs)
    ys = np.array(ys)
    
    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u/2
    pos_y = ys[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    
    fig, ax = plt.subplots()
    ax.plot(xs,ys, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    #ax.set_xlabel('FEATURE_%i' %f1)
    #ax.set_ylabel('FEATURE_%i' %f2)
    plt.show()
    return xs, ys

# opt_trajectory() takes a maze name, then solves the policy and plots the
# optimal path through the maze. Returns the path. ONLY WORKS for deterministic
# mazes!
def opt_trajectory(maze):
    P, R = get_maze_MDP(maze)
    v, pi = SolveMDP(P, R, 0.98, 1e-10, True, 'max')
    
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    reward = -0.1/(l*l)
    
    xs = [obs[0]]
    ys = [-obs[1]]
    done = False
    
    while not done:
        # find current state and action
        ogc = int(obs[0] + obs[1]*l)
        #print(ogc)
        a = int(pi[ogc])
        
        obs, reward, done, info = env.step(a)
        #print(done)
        xs.append(obs[0])
        ys.append(-obs[1])

    xs = np.array(xs)
    ys = np.array(ys)
    
    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u/2
    pos_y = ys[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 
    
    fig, ax = plt.subplots()
    ax.plot(xs,ys, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    #ax.set_xlabel('FEATURE_%i' %f1)
    #ax.set_ylabel('FEATURE_%i' %f2)
    plt.show()
    return xs, ys

# opp_action() returns the opposite action as input action
def opp_action(a):
    if a == 0:
        return 1
    elif a == 1:
        return 0
    elif a == 2:
        return 3
    elif a == 3:
        return 2
    
# get_maze_MDP() takes a maze string name, and returns two matrices, P and R, 
# which describe the MDP of the maze
def get_maze_MDP(maze):
    # initialize maze
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    
    # initialize matrices
    a = 4
    P = np.zeros((a, l*l, l*l))
    R = np.zeros((a, l*l))
    
    # store clusters seen and cluster/action pairs seen in set
    c_seen = set()
    ca_seen = set()
    
    # initialize env, set reward of original
    obs = env.reset()
    ogc = int(obs[0] + obs[1]*l)
    reward = -0.1/(l*l)
    
    while len(ca_seen) < 4*l*l:
        # update rewards for new cluster
        if ogc not in c_seen:
            for i in range(a):
                R[i, ogc] = reward
            c_seen.add(ogc)
        
        stop = False
        for i in range(a):
            if (ogc, i) not in ca_seen:
                ca_seen.add((ogc, i))
                #print(len(ca_seen))
                new_obs, reward, done, info = env.step(i)
                ogc_new = int(new_obs[0] + new_obs[1]*l)
                # update probabilities
                P[i, ogc, ogc_new] = 1
                if ogc != ogc_new:
                    P[opp_action(i), ogc_new, ogc] = 1
                    ca_seen.add((ogc_new, opp_action(i)))
                    #print(len(ca_seen))
                ogc = ogc_new
                
                if done:
                    obs = env.reset()
                    ogc = int(obs[0] + obs[1]*l)
                
                stop = True
            if stop:
                break
        
        # if all seen already, take random step 
        if not stop:
            action = env.action_space.sample()
            new_obs, reward, done, info = env.step(action)
            ogc = int(new_obs[0] + new_obs[1]*l)
            #print('trying random action', ogc)

    return P, R


# plot_paths() takes a dataframe with 'FEATURE_1' and 'FEATURE_2', and plots
# the first n paths (by ID). returns nothing
def plot_paths(df, n): 
    fig, ax = plt.subplots()
    
    for i in range(n):
        x = df.loc[df['ID']==i]['FEATURE_0']
        y = -df.loc[df['ID']==i]['FEATURE_1']
        xs = np.array(x)
        ys = np.array(y)
        
        u = np.diff(xs)
        v = np.diff(ys)
        pos_x = xs[:-1] + u/2
        pos_y = ys[:-1] + v/2
        norm = np.sqrt(u**2+v**2) 
        
        ax.plot(xs,ys, marker="o")
        ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    
    plt.show()
    return
        
# Initialize the "maze" environment
# =============================================================================
# =============================================================================
# env = gym.make(mazes[4])
# obs = []
# 
# # first point for ID
# observation = env.reset()
# offset = np.array((random.random(), random.random()))
# # #env.state = offset
# # #print(env.state)
# obs.append(observation+offset)
# 
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

# =============================================================================
