#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:47:27 2020

@author: janiceyang
"""
#################################################################
# Load Libraries
import numpy as np
import pandas as pd
import random
#from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from scipy.spatial import distance
#################################################################

# f() takes in x (the features T1 to E), the action pair u = (u1,u2) (u1=1 when 
# RTI on, u2 = 1 when PI on; 0 otherwise for both), and the time horizon t, and 
# computes the transition from x_0 to x_t having taken action u1 u2. Returns 
# the new features x_t (T1, T2... V, E)
def f(x, u, t, dt = 0.0005):
    T1, T2, Ts1, Ts2, V, E = x
    u1, u2 = u
    # defining action parameters
    # for RTI
    if u1 == 1:
        e1 = 0.7
    else:
        e1 = 0
    # for PI
    if u2 == 1:
        e2 = 0.3
    else:
        e2 = 0
    
    # defining parameters
    l1 = 10000
    d1 = 0.01
    k1 = 8e-7
    l2 = 31.98
    d2 = 0.01
    f = 0.34 # confirm this doesn't interfere with function name
    k2 = 1e-4
    delt = 0.7
    m1 = 1e-5
    m2 = 1e-5
    NT = 100
    c = 13
    rho1 = 1
    rho2 = 1
    lE = 1
    bE = 0.3
    Kb = 100
    dE = 0.25
    Kd = 500
    deltE = 0.1
    
    n=int(t/dt)
    # iterating through small x_t updates 
    for i in range(n):
        T1_new = T1 + (l1 - d1*T1 - (1-e1)*k1*V*T1)*dt
        T2_new = T2 + (l2 - d2*T2 - (1-f*e1)*k2*V*T2)*dt
        Ts1_new = Ts1 + ((1-e1)*k1*V*T1 - delt*Ts1 - m1*E*Ts1)*dt
        Ts2_new = Ts2 + ((1-f*e1)*k2*V*T2 - delt*Ts2 - m2*E*Ts2)*dt
        E_new = E + (lE + (bE*(Ts1+Ts2)*E)/((Ts1+Ts2)+Kb) - \
                     (dE*(Ts1+Ts2)*E)/((Ts1+Ts2)+Kd) - deltE*E)*dt
        V_new = V + ((1-e2)*NT*delt*(Ts1+Ts2) - c*V - \
                     ((1-e1)*rho1*k1*T1 + (1-f*e1)*rho2*k2*T2)*V)*dt
        
        # update variables
        T1 = T1_new
        T2 = T2_new
        Ts1 = Ts1_new
        Ts2 = Ts2_new
        E = E_new
        V = V_new
        #print((T1, T2, Ts1, Ts2, V, E))
        
    return (T1, T2, Ts1, Ts2, V, E)

# c_a() takes in a tuple of features x, and a tuple of actions u = (u1, u2),
# and returns the action-dependent cost of the current state-action pair
def c_a(x, u):
    T1, T2, Ts1, Ts2, V, E = x
    u1, u2 = u
    
    # defining parameters
    Q = 0.1
    R1 = 20000
    R2 = 2000
    S = 1000
    
    # defining action parameters
    # for RTI
    if u1 == 1:
        e1 = 0.7
    else:
        e1 = 0
    # for PI
    if u2 == 1:
        e2 = 0.3
    else:
        e2 = 0
    
    # calculating cost
    c = Q*V + R1*(e1**2) + R2*(e2**2) - S*E
    return c


# c_r() takes in a tuple of features x, and returns the action-independent cost 
# of the current state
def c_r(x):
    T1, T2, Ts1, Ts2, V, E = x
    
    # defining parameters
    Q = 0.1
    S = 1000
    
    # calculating cost 
    c = Q*V - S*E
    return c

'''
# healthy equilibrium given
#x = (967839, 621, 76, 6, 415, 353108)

# unhealthy equilibrium given
x = (163573, 5, 11945, 46, 63919, 24)
#x = (100000, 3, 5000, 10, 60000, 10)

u = (1, 0)
t = 1

for i in range(100):
    u = (np.random.randint(0, 2), np.random.randint(0, 2))
    x = f(x, u, 5)
    print(u)
    print(x)

print(x)

print('cost-action:', c_a(x, u))
print('cost-state:', c_r(x))
'''


# fitted_Q() trains K functions Q1 to QK that determine the optimal strategy
# x_df is a dataframe of the form ['x_t', 'u', 'x_t2', 'c'] for each one-step transition
# returns the last iteration QK, and a function policy that takes a state and
# outputs the optimal action
def fitted_Q(K, x_df, gamma):
    
    x_df = x_df.copy(deep=True)
    # initialize storage and actions
    Qs = []
    Us = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # create the first Q1 function
    class Q:
        def predict(self, array): 
            x = array[0][:6]
            u = array[0][6:]
            return c_a(x, u)
    Q1 = Q()
    Qs.append(Q1)
    
    # create X using x_t and u
    # select (x_t, u) pair as training
    X = x_df['x_t'].apply(pd.Series).merge(x_df['u'].apply(pd.Series), \
                left_index = True, right_index = True) 
    print('New training features')
    print(X)
    
    
    #bar = tqdm(range(K))
    bar = range(K)
    #creating new Qk functions
    for i in bar:
        #print('starting')
        # THIS IS the slowest part!! going through each line and predicting
        # 4 things (especially slow for random forest)
        # create y using Qk-1 and x_t2
        y = x_df.apply(lambda x: x.c + gamma*min([Qs[-1].predict([f]) \
                                for f in [np.array(x.a0), np.array(x.a1), \
                                          np.array(x.a2), np.array(x.a3)]]), axis=1)
                                          
        #y = x_df.apply(lambda x: x.c + gamma*min([#Qs[-1].predict(np.array(x.a0)), \
                                                  #Qs[-1].predict(np.array(x.a1)), \
                                                  #Qs[-1].predict(np.array(x.a2)), \
                                    #Qs[-1].predict(np.array(x.a3).reshape(1, -1))]), axis=1)
        #print(y)
        
        # train the actual Regression function as Qk
        #regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
        #regr = LinearRegression().fit(X, y)
        #regr = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
        regr = ExtraTreesRegressor(n_estimators=50).fit(X,y)
        Qs.append(regr)
        
    
    QK = Qs[-1]
    
    p = policy()
    p.fit(QK)
        
    return QK, p, Qs


# get_policy() takes in a QK function (last learned function from fitted-Q), 
# and returns a function policy(). policy() in a state x, and returns the 
# action that results in the minimum cost, calculated using the most updated 
# fitted_Q QK
class policy:
    def __init__(self):
        self.QK = None
        
    def fit(self, 
            QK): # model, the latest fitted_Q 
        self.QK = QK
    
    # pred() takes a state x and predicts the optimal action
    def get_action(self,
             x): 
        Us = [(0, 0), (0, 1), (1, 0), (1, 1)]
        i = np.argmin([self.QK.predict([np.array([a for b in \
                                            [x, u] for a in b])]) \
                                        for u in Us])
        return Us[i]
    


# gen() takes N, the number of patients, and T, the number of timesteps,
# and a float r and a function policy, where r is the percentage of
# datapoints that will follow random actions, while the rest follows policy. 
# returns a dataframe of all the transitions in the form ['x_t', 'u', 'x_t+1', 'c', 'a0'...]
def gen(N, 
        T, 
        r, 
        cost, # str: 'c_a' or 'c_r' depending on reward dependence
        p): # a trained policy model
    # starting in non-healthy steady state
    x_init = (163573, 5, 11945, 46, 63919, 24)
    
    transitions = []
    # creating patient transitions
    for i in range(N):
        x = x_init
        for t in range(T):
            # smaller than r means takes random action, else optimal policy
            if random.random() <= r: 
                u = (np.random.randint(0, 2), np.random.randint(0, 2))
            else:
                u = p.get_action(x)
            xt = f(x, u, 5) # simulating 5 day change
            if cost == 'c_a':
                c = c_a(x, u)
            if cost == 'c_r':
                c = c_r(x)
            transitions.append([x, u, xt, c])
            # update new x
            x = xt
            #print('t', t, 'x', x)
    
    df = pd.DataFrame(transitions, columns=['x_t', 'u', 'x_t2', 'c'])
    # create flattened feature vectors for each action: [(x_t2, u)]
    Us = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i in range(4):
        df['a%s'%i] = df.apply(lambda x: tuple([a for b in \
                                    [x.x_t2, Us[i]] for a in b]), axis=1)
    return df


# createSamples() creates a list of samples in a format that can directly
# be integrated into the main MDP algorithm. 
# returns a dataframe in the form ['ID', 'TIME', ...features..., 'RISK', 'ACTION]
def createSamples(N,
                  T,
                  r,
                  cost, # 'c_a', 'c_r', or 'dist' 
                  thresh, # if cost == 'dist', thresh is threshold around healthy & unhealthy points
                  p=None): 
    
    # starting in non-healthy steady state
    unhealthy = (163573, 5, 11945, 46, 63919, 24)
    healthy = (967839, 621, 76, 6, 415, 353108)
    
    transitions = []
    # creating patient transitions
    for i in range(N):
        ID = i
        x = unhealthy
        for t in range(T):
            # smaller than r means takes random action, else optimal policy
            TIME = t
            if random.random() <= r: 
                u = (np.random.randint(0, 2), np.random.randint(0, 2))
            else:
                u = p.get_action(x)
            
            xt = f(x, u, 5) # simulating 5 day change
            
            if cost == 'dist': 
                if distance.euclidean(x, unhealthy) < thresh:
                    c = -1
                elif distance.euclidean(x, healthy) < thresh:
                    c = 1
                else: 
                    c = 0
            if cost == 'c_a':
                c = c_a(x, u)
            if cost == 'c_r':
                c = c_r(x)
            a = convert(u)
            transitions.append([ID, TIME, x, a, c])
            # update new x
            x = xt
            #print('t', t, 'x', x)
    #print(transitions)
    
    df = pd.DataFrame(transitions, columns=['ID', 'TIME', 'x', 'ACTION', 'RISK'])
    #print(df)
    features = df['x'].apply(pd.Series)
    features = features.rename(columns = lambda x : 'FEATURE_' + str(x))
    
    df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
    
    return df_new

# convert() takes u, either an integer or an action tuple, and returns the
# corresponding other equivalent
def convert(u):
    Us = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if type(u) == int: 
        return Us[u]
    else:
        return Us.index(u)
        
#################################################################
# Running the experiment
'''
policies = []
Q_opt = []
all_Qs = []
# creating first 6000, completely random
df_all = gen(30, 60, 1, 'c_a', None)
Q, p, Qs = fitted_Q(10, df_all, 0.98) #change 10 to 400
policies.append(p)
Q_opt.append(Q)
all_Qs.append(Qs)


# creating new samples via 15% random, 85% optimal policy from ALL 
# prior samples
for i in range(4):
    print('starting set ', i)
    df2 = gen(30, 60, 0.15, 'c_a', p)
    df_all = pd.concat([df_all, df2], ignore_index=True)
    Q, p, Qs = fitted_Q(10, df_all, 0.98) #change 10 to 400
    policies.append(p)
    Q_opt.append(Q)
    all_Qs.append(Qs)
    
    # displaying the optimal policies here
    df_test = gen(1, 60, 0, 'c_a', p)
    print(df_test['u'])
    print(df_test[['x_t', 'c']])

'''
    