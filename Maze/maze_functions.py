#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:57:14 2020

@author: janiceyang
"""

import sys
import numpy as np
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

def createSamples(N, T, maze):
    pass

# Initialize the "maze" environment
env = gym.make("maze-sample-3x3-v0")
obs = []

# first point for ID
observation = env.reset()
offset = np.array((random.random(), random.random()))
env.state = offset
print(env.state)
obs.append(observation+offset)
for _ in range(10):
    
    #env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print(observation, reward, done, info)
    obs.append(observation+offset)
      
    if done:
      observation = env.reset()
      # break?? for new ID?
#env.close()