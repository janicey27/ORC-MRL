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
#import gym
import pickle

#import sys
#sys.path.append('/Users/janiceyang/Dropbox (MIT)/ORC UROP/Opioids/Algorithm/')
from MDPtools import SolveMDP
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


# save_all_errors() takes a list of models and policies, and saves the 4
# arrays corresponding to testing and training errors of both
def save_all_errors(models, policies, Ns, maze, df_test):
    model_training_accuracies = []
    model_testing_accuracies = []
    policy_training_accuracies = []
    policy_testing_accuracies = []
    
    n = len(Ns)
    
    # for each n of this set:
    for i in range(n):
        m = models[i]
        p = policies[i]
        
        m_train_acc = policy_accuracy(m, maze, m.df)
        model_training_accuracies.append(m_train_acc)
        
        p_train_acc = fitted_Q_policy_accuracy(p, maze, m.df)
        policy_training_accuracies.append(p_train_acc)
        
        m_test_acc = policy_accuracy(m, maze, df_test)
        model_testing_accuracies.append(m_test_acc)
        
        p_test_acc = fitted_Q_policy_accuracy(p, maze, df_test)
        policy_testing_accuracies.append(p_test_acc)
        
        print('N=', Ns[i], 'complete')
        
    return model_training_accuracies, policy_training_accuracies, model_testing_accuracies, policy_testing_accuracies
        

# policy_accuracy() takes a trained model and a maze, compares every line of 
# the original dataset to the real optimal policy and the model's optimal policy, 
# then returns the percentage correct from the model
def policy_accuracy(m, maze, df):
    if m.v is None:
        m.solve_MDP()
    
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


# load_and_avg_all_acc() takes a list of Ns, an int s indicating the number of 
# sets to load, and returns the averaged accuracies for model and policy
# testing and training
def load_and_avg_all_acc(Ns, s):
    m1s = np.zeros(len(Ns))
    m2s = np.zeros(len(Ns))
    p1s = np.zeros(len(Ns))
    p2s = np.zeros(len(Ns))
    for i in range(s):
    
        m1 = pickle.load(open('r{}_m_train_acc.sav'.format(i), 'rb'))
        m2 = pickle.load(open('r{}_m_test_acc.sav'.format(i), 'rb'))
        p2 = pickle.load(open('r{}_p_test_acc.sav'.format(i), 'rb'))
        p1 = pickle.load(open('r{}_p_train_acc.sav'.format(i), 'rb'))
    
        m1s = m1s + m1
        m2s = m2s + m2
        p1s = p1s + p1
        p2s = p2s + p2
    
    m1 = m1s/s
    m2 = m2s/s
    p1 = p1s/s
    p2 = p2s/s
    
    return m1, p1, m2, p2
    
    
# plot_all() takes 4 arrays of training and testing accuracies, and plots them
def plot_all_acc(m_train, p_train, m_test, p_test, Ns):
    fig1, ax1 = plt.subplots()
    ax1.plot(Ns, m_train, label= "Model Train Accuracy")
    ax1.plot(Ns, p_train, label = "Fitted_Q Train Accuracy")
    ax1.plot(Ns, m_test, label= "Model Test Accuracy")
    ax1.plot(Ns, p_test, label = "Fitted_Q Test Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('N training data size')
    ax1.set_ylabel('Accuracy %')
    ax1.set_title('Accuracy comparison')
    ax1.legend()
    plt.show()
    return


# value_diff() takes a list of models, calculates the difference between values
# |v_policy/algo - v_opt*|. v_policy/algo is found by randomly generating K 
# points in the starting cell, simulating over t_max steps, and taking the avg
# over these K trials. v_opt is the value 
def value_diff(models, policies, Ns, maze, K, t_max): 
    # first calculate v_opt for this particular maze and t_max steps
    
    # then for each model and policy, run through the K trials and return
    # an array of differences corresponding to each N 
    pass


if __name__ == "__main__":
    # Importing the relevant matrices and values
    P = pickle.load(open('5x5_P.sav', 'rb'))
    R = pickle.load(open('5x5_Reward.sav', 'rb'))
    
    true_v = pickle.load(open('5x5_true_v.sav', 'rb'))
    true_pi = pickle.load(open('5x5_true_pi.sav', 'rb'))
    
    df_full = pd.read_csv('df_test.csv')
    df_full.loc[df_full['ACTION']=='None', 'ACTION'] = 4
    df_full['ACTION'] = pd.to_numeric(df_full['ACTION'], downcast='integer')
    df_full.loc[df_full['ACTION']==4, 'ACTION'] = 'None'
    df_test = df_full
    #################################################################
    
    
    #################################################################
    # Import Models and Policies -- must have all the models in this folder!
    Ns = [5, 10, 20, 30, 35, 40, 45, 50, 70, 90, 110, 130, 150, 170, 200]
    #Ns = [2, 3, 5, 10, 20, 30, 50, 70, 90, 110, 130, 150, 170, 200]
    #Ns = [5, 10, 20, 30, 40, 50, 70]
    
    for s in range(30):
        models = [] # our trained models
        policies = [] # fitted_Q policies
        
        for n in Ns: 
            model_name = 'round_{}_model_N={}.sav'.format(s, n)
            policy_name = 'round_{}_fqpolicy_N={}.sav'.format(s, n)
            
            m = pickle.load(open(model_name, 'rb'))
            p = pickle.load(open(policy_name, 'rb'))
            
            models.append(m)
            policies.append(p)
            
        print('Models Loaded Successful')
        
        # saving policy and model accuracies
        m_train, p_train, m_test, p_test = save_all_errors(models, policies, Ns, mazes[4], df_test)
        
        
        pickle.dump(m_train, open('r{}_m_train_acc.sav'.format(s, n), 'wb'))
        pickle.dump(p_train, open('r{}_p_train_acc.sav'.format(s, n), 'wb'))
        pickle.dump(m_test, open('r{}_m_test_acc.sav'.format(s, n), 'wb'))
        pickle.dump(p_test, open('r{}_p_test_acc.sav'.format(s, n), 'wb'))
        
        # saving value differences
        print('set ', s, 'completed')
    #################################################################
