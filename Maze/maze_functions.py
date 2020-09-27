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
from MDPtools import SolveMDP

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

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from tqdm import tqdm
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
             

# opt_model_trajectory() takes a trained model, the maze used to train this model, and 
# plots the path of the optimal solution through the maze. returns the path
def opt_model_trajectory(m, maze, alpha, min_action_obs=0, min_action_purity=0):
    #if m.v is None:
        #m.solve_MDP()
    m.solve_MDP(alpha, min_action_obs, min_action_purity)
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    reward = -0.1/(l*l)
    
    xs = [obs[0]]
    ys = [-obs[1]]
    done = False

    offset = np.array((random.random(), -random.random()))
    point = np.array((obs[0], -obs[1])) + offset

    while not done:
        # find current state and action
        s = m.m.predict(point.reshape(1, -1))
        #print(s)
        a = int(m.pi[s])
        #print(a)
        
        obs, reward, done, info = env.step(a)
        
        offset = np.array((random.random(), -random.random()))
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
    plt.ylim(-l+0.8, 0.2)
    plt.xlim(-.2, l-0.8)
    plt.show()
    return xs, ys


# def policy_trajectory() takes a policy, a maze, and plots the optimal
# trajectory of the policy through the maze, for a total of n steps. 
# Can use with fitted_Q policy etc.
def policy_trajectory(policy, maze, n=50, rand=True):
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    reward = -0.1/(l*l)
    
    
    offset = np.array((random.random(), -random.random()))
    point = list(np.array((obs[0], -obs[1])) + offset)

    if rand:
        xs = [point[0]]
        ys = [point[1]]
    else:
        xs = [obs[0]]
        ys = [-obs[1]]
    
    done = False
    i = 0
    
    while not done:
        # find relevant action
        #print(point)
        a = policy.get_action(point)
        #print(a)
        obs, reward, done, info = env.step(a)
        
        offset = np.array((random.random(), -random.random()))
        point = list(np.array((obs[0], -obs[1])) + offset)
        
        if rand:
            xs.append(point[0])
            ys.append(point[1])
        else:
            xs.append(obs[0])
            ys.append(-obs[1])
        
        i += 1
        if i == n:
            done = True
    
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
    if rand:
        plt.ylim(-l-0.2, 0.2)
        plt.xlim(-.2, l+0.2)
    else:
        plt.ylim(-l+0.8, 0.2)
        plt.xlim(-.2, l-0.8)
    plt.show()
    return xs, ys
        
    


# policy_accuracy() takes a trained model and a maze, compares every line of 
# the original dataset to the real optimal policy and the model's optimal policy, 
# then returns the percentage correct from the model
def policy_accuracy(m, maze, df):
    if m.v is None:
        m.solve_MDP()
        
    # finding the true optimal: 
    P, R = get_maze_MDP(maze)
    true_v, true_pi = SolveMDP(P, R, 0.98, 1e-10, True, 'max')
    
    correct = 0
    # iterating through every line and comparing 
    for index, row in df.iterrows():
        # predicted action: 
        s = m.m.predict(np.array(row[2:2+m.pfeatures]).reshape(1,-1))
        #s = m.df_trained.iloc[index]['CLUSTER']
        a = m.pi[s]
        
        # real action:
        a_true = true_pi[row['OG_CLUSTER']]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct/total

# fitted_Q_policy_accuracy() takes a policy given by fitted_Q, the maze, 
# and the original dataframe, compares every line of the original dataset to
# the real optimal policy and the fitted_Q's optimal policy, then returns the
# percentage correct from fitted_Q 
def fitted_Q_policy_accuracy(policy, maze, df):
        
    # finding the true optimal: 
    P, R = get_maze_MDP(maze)
    true_v, true_pi = SolveMDP(P, R, 0.98, 1e-10, True, 'max')
    
    correct = 0
    # iterating through every line and comparing 
    for index, row in df.iterrows():
        # predicted action: 
        a = policy.get_action(list(row[2:4]))
        
        # real action:
        a_true = true_pi[row['OG_CLUSTER']]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct/total



# opt_maze_trajectory() takes a maze name, then solves the policy and plots the
# optimal path through the maze. Returns the path. ONLY WORKS for deterministic
# mazes!
def opt_maze_trajectory(maze):
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
# which describe the MDP of the maze (includes a sink node)
def get_maze_MDP(maze):
    # initialize maze
    env = gym.make(maze)
    obs = env.reset()
    l = env.maze_size[0]
    
    # initialize matrices
    a = 4
    P = np.zeros((a, l*l+1, l*l+1))
    R = np.zeros((a, l*l+1))
    #P = np.zeros((a, l*l, l*l))
    #R = np.zeros((a, l*l))
    
    # store clusters seen and cluster/action pairs seen in set
    c_seen = set()
    ca_seen = set()
    
    # initialize env, set reward of original
    obs = env.reset()
    ogc = int(obs[0] + obs[1]*l)
    reward = -0.1/(l*l)
    
    while len(ca_seen) < (4*l*l-4):
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
                #print('updated', ogc, ogc_new, done)
                if not done:
                    if ogc != ogc_new:
                        P[opp_action(i), ogc_new, ogc] = 1
                        #print('updated', ogc_new, ogc)
                        ca_seen.add((ogc_new, opp_action(i)))
                    #print(len(ca_seen))
                ogc = ogc_new
                #print('new ogc', ogc, done)
                
                if done:
                    # set next state to sink
                    for i in range(a):
                        P[i, ogc_new, l*l] = 1
                        P[i, l*l, l*l] = 1
                        R[i, l*l] = 0
                        R[i, ogc_new] = 1
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
            if done:
                obs = env.reset()
                ogc = int(obs[0] + obs[1]*l)

    return P, R


# get_maze_transition_reward() takes a maze name, and returns the transition function
# in the form of f(x, u) = x'. State, action, gives next state. Takes into account
# sink node, and stays there. Also returns reward function r(x) that takes a state
# and returns the reward
def get_maze_transition_reward(maze):
    
    P, R = get_maze_MDP(maze)
    l = int((R.size/4-1)**0.5)
    
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
    
    return f, r
    


# plot_paths() takes a dataframe with 'FEATURE_1' and 'FEATURE_2', and plots
# the first n paths (by ID). returns nothing
def plot_paths(df, n): 
    fig, ax = plt.subplots()
    
    for i in range(n):
        x = df.loc[df['ID']==i]['FEATURE_0']
        y = df.loc[df['ID']==i]['FEATURE_1']
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



# fitted_Q() trains K functions Q1 to QK that determine the optimal strategy
# x_df is a dataframe of the form ['ID', 'TIME', features, 'ACTION', 'RISK']
# for each one-step transition returns the last iteration QK, and a function \
# policy that takes a state and outputs the optimal action
def fitted_Q(K, # number of iterations
             x_df, # dataframe 
             gamma, # decay factor
             pfeatures, # number of features in dataframe
             actions, # list of action possibilities
             f, # transition function 
             reward_c, # reward function
             take_max = True, # True if max_cost is good, otherwise false
             regression = 'LinearRegression' # str: type of regression
             ):
    
    x_df = x_df.copy(deep=True)
    #x_df = x_df.loc[x_df['']]
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
    x_df['x_t2'] = x_df.apply(lambda x: list(f(tuple(x[2:2+pfeatures]), x.ACTION, False)), \
                                          axis=1)
        
        
    for i in range(len(actions)):
        x_df['a%s'%i] = x_df.apply(lambda x: x.x_t2+[actions[i]], axis=1)
    action_names = ['a%s'%i for i in range(len(actions))]
    
    print(x_df, flush=True)
    # create X using x_t and u
    # select (x_t, u) pair as training
    # setting 'None' action as action 4
    X = x_df.iloc[:, 2:3+pfeatures]
    X.loc[X['ACTION']=='None', 'ACTION'] = 4
    
    
    # maybe trying to put action as a tuple of (0, 1) or 1-hot to help it learn better...?
    #X = x_df[['FEATURE_0', 'FEATURE_1']].merge(x_df['ACTION'].apply(pd.Series), \
                #left_index = True, right_index = True) 
    
    print('New training features', flush=True)
    print(X, flush=True)
    
    
    bar = tqdm(range(K))
    #bar = range(K)
    #creating new Qk functions
    for i in bar:
        # create y using Qk-1 and x_t2
        # non-DP
        if take_max: 
            y = x_df.apply(lambda x: x.RISK + gamma*max([Q_new.predict([g]) \
                                for g in [x[a] for a in action_names]]), axis=1)
        else:
            y = x_df.apply(lambda x: x.RISK + gamma*min([Q_new.predict([g]) \
                                for g in [x[a] for a in action_names]]), axis=1)
        
        
        
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
        #print(y, flush=True)
        #print(np.unique(y), flush=True)
        #print(y)
        
        # train the actual Regression function as Qk
        #regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
        if regression ==  'LinearRegression':
            regr = LinearRegression().fit(X, y)
        if regression == 'RandomForest':
            regr = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
        if regression == 'ExtraTrees':
            regr = ExtraTreesRegressor(n_estimators=50).fit(X,y.ravel())
        if regression == 'SGDRegressor':
            regr = SGDRegressor().fit(X, y.ravel())
        #Qs.append(regr)
        Q_new = regr
        #print('memo size', len(memo), 'used', mu, flush=True)
        
    
    #QK = Qs[-1]
    QK = Q_new
    
    p = policy(actions, take_max)
    p.fit(QK)
        
    return QK, p, x_df


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
    
    
# value_diff() takes a list of models, and the real transitions calculates the difference between values
# |v_policy/algo - v_opt*|. v_policy/algo is found by randomly generating K 
# points in the starting cell, simulating over t_max steps, and taking the avg
# over these K trials. v_opt is the value taking the optimal policy
def value_diff(models, Ns, K, t_max, P, R, f, r): 
    # first calculate v_opt for this particular maze and t_max steps
    v_opt = 0
    s = 0
    true_v, true_pi = SolveMDP(P, R, prob='max')
    for t in range(t_max):
        v_opt += R[0, s]
        #print(R[0, s], v_opt)
        a = true_pi[s]
        s_new = P[a, s].argmax()
        s = s_new
        #print(s)
    
    # then for each model and policy, run through the K trials and return
    # an array of differences corresponding to each N 
    n = len(Ns)
    
    v_alg = []
    #v_policy = []
    # for each n of this set:
    for i in range(n):
        print('Round N=', Ns[i])
        m = models[i]
        #p = policies[i]
        
        # calculating average value for this model and policy
        if m.pi is None:
            print('resolved model')
            m.solve_MDP()
        
        model_vs = []
        #policy_vs = []
        
        # initialize a list of random starting points (or not, since we won't be able to
        # keep the rest of the transitions the same randomness anyway)
        for k in range(K):
            vm_estim = 0 # initializing model value estimate
            vp_estim = 0 # initializing policy value estimate
            
            x_model = np.array((random.random(), -random.random()))
            #x_policy = np.array((random.random(), -random.random()))
            
            # estimate value for model & policy
            vm_estim += r(x_model)
            #vp_estim += r(x_policy)
            
            for t in range(t_max):
                # predict action and upate value for model
                if x_model[0] == None:
                    a = 0
                else:
                    s = int(m.m.predict([x_model]))
                    a = m.pi[s]
                x_model_new = f(x_model, a)
                vm_estim += r(x_model_new)
                #print('new model state', x_model_new, 'reward', r(x_model_new))
                x_model = x_model_new
                
                '''
                # predict action and update value for policy
                if x_policy[0] == None:
                    u = 0
                else:
                    u = p.get_action(list(x_policy))
                x_policy_new = f(x_policy, u)
                vp_estim += r(x_policy_new)
                #print('new policy state', x_policy_new, 'reward', r(x_policy_new))
                x_policy = x_policy_new
                '''
            
            # append the total value from this trial
            model_vs.append(vm_estim)
            '''
            policy_vs.append(vp_estim)
            print('final model value', vm_estim)
            print('final policy value', vp_estim)
            '''
        
        # average values of all trials for this model/policy
        model_v = np.mean(model_vs)
        v_alg.append(model_v)
        '''
        policy_v = np.mean(policy_vs)
        v_policy.append(policy_v)
        '''
        #print('model avg for this trial', model_v)
        #print('policy avg for this trial', policy_v)
    
    # calculate differences between values and optimal
    v_alg_diff = abs(v_alg - v_opt) # np array minus value for all elements --
                                    #TODO: Check these!!!
    #print('alg array', v_alg, 'v_opt', v_opt)
    #print('difference array', v_alg_diff)
    '''
    v_policy_diff = abs(v_policy - v_opt)
    '''
    
    return v_alg_diff
        
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
