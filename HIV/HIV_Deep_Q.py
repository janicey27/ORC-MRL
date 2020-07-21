# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:19:34 2020

@author: omars
"""

#%% Set Working Directory
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import gym
import sys


from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from HIV_functions import f, c_a, convert, J

#os.chdir("C:/Users/omars/Desktop/opioids/Opioids/Algorithm/")
import sys
sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')

from MDPtools import *
from model import MDP_model
#from HIV_functions import *
from clustering import *
from testing import *


#%% Set Parameters
clustering = 'Agglomerative'
n_clusters = None
distance_threshold = 0.5
random_state = 0
h = 2
max_k = 50
cv = 5
th = 0
classification = 'DecisionTreeClassifier'
thresh = 0 # threshold for dist when deciding risk
ENV_NAME = "CartPole-v1"
GAMMA = 0.98
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.999

#%% Optimal Solver

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.observation_space = observation_space

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        state = np.reshape(state, [1, self.observation_space])
        q_values = self.model.predict(state)
        #print('predicted success')
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            #print('reward', reward)
            q_update = reward
            if not terminal:
                state_next = np.reshape(state_next, [1, self.observation_space])
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            state = np.reshape(state, [1, self.observation_space])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    def total_cost(self):
        x = [163573, 5, 11945, 46, 63919, 24]
        T = 200
        gamma = 0.98
        total = 0
        a = self.act(x)
        for i in range(T):
            total += (gamma**i)*c_a(x, a)
            x = f(x, a, 5)
            a = self.act(x)
        return total

#%% Train Deep Q Learning

env = gym.make(ENV_NAME)
#score_logger = ScoreLogger(ENV_NAME)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
dqn_solver = DQNSolver(observation_space, action_space)
run = 0
while True:
    run += 1
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
        step += 1
        #env.render()
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        state_next = np.reshape(state_next, [1, observation_space])
        dqn_solver.remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
            #score_logger.add_score(step, run)
            break
        dqn_solver.experience_replay()
        
        
#%% HIV Train Deep Q Learning


#score_logger = ScoreLogger(ENV_NAME)
observation_space = 6
action_space = 4
dqn_solver = DQNSolver(observation_space, action_space)
run = 0
while True:
    run += 1
    state = np.array([163573, 5, 11945, 46, 63919, 24])
    #state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
        step += 1
        #env.render()
        action = dqn_solver.act(state)
        state_next = f(state, action, 5)
        #print(state_next)
        reward = -c_a(state, action) 
        terminal = ([967839, 621, 76, 6, 415, 353108]==[round(n) for n in state_next])
        state_next = np.array(state_next)
        dqn_solver.remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print('Healthy state reached!!')
            break
        dqn_solver.experience_replay()
        if step%100 == 0:
            print('At step:', step)
            print('Converging cost of current policy:', dqn_solver.total_cost())

#%% Cartpole Cool Visualization
env = gym.make('CartPole-v0')
env.reset()
for i in range(1000):
    env.render()
    features, reward, done,  _ = env.step(env.action_space.sample()) # take a random action
    if done and i > 300:
        break

env.close()
#%% Cartpole Generation
n_test = 100
n_steps = 1000
l_list = []
p = 0.7
for test in range(n_test):
    print(test)
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    for step in range(n_steps):
        #env.render()
        if random.random() <= p:
            action = dqn_solver.act(state)
        else:
            action = 1 - dqn_solver.act(state)
        action = env.action_space.sample()
        state_next, reward, done, info = env.step(action)
        if done:
            l_list.append([test, step] + list(state[0]) + [action, -reward])
            env.reset()
            break
        else:
            l_list.append([test, step] + list(state[0]) + [action, reward])
        state = np.reshape(state_next, [1, observation_space])
    #env.close()

pfeatures=4
col_names = ['ID', 'TIME'] + ['FEATURE_' + str(i) for i in range(pfeatures)] + ['ACTION', 'RISK']
df = pd.DataFrame(l_list, columns=col_names)

#%% Train MDP
m = MDP_model()
m.fit(df, # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
    pfeatures, # int: number of features
    h, # int: time horizon (# of actions we want to optimize)
    max_k, # int: number of iterations
    distance_threshold, # clustering diameter for Agglomerative clustering
    cv, # number for cross validation
    th, # splitting threshold
    classification, # classification method
    {'random_state':0},
    clustering,# clustering method from Agglomerative, KMeans, and Birch
    n_clusters, # number of clusters for KMeans
    random_state,
    plot=True)

#%% Testing set
n_test = 100
n_steps = 1000
l_list = []
p = 0.7
for test in range(n_test):
    print(test)
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    for step in range(n_steps):
        #env.render()
        if random.random() <= p:
            action = dqn_solver.act(state)
        else:
            action = 1 - dqn_solver.act(state)
        #action = env.action_space.sample()
        state_next, reward, done, info = env.step(action)
        if done:
            l_list.append([test, step] + list(state[0]) + [action, -reward])
            env.reset()
            break
        else:
            l_list.append([test, step] + list(state[0]) + [action, reward])
        state = np.reshape(state_next, [1, observation_space])
    #env.close()

pfeatures=4
col_names = ['ID', 'TIME'] + ['FEATURE_' + str(i) for i in range(pfeatures)] + ['ACTION', 'RISK']
df_test = pd.DataFrame(l_list, columns=col_names)

#%% Solve MDP

v, pi = m.solve_MDP()

#%% Testing error
test_error = testing_value_error(df_test, m.df_trained, m.m, pfeatures,relative=False,h=h)
print(test_error)