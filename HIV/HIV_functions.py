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
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
#################################################################

# f() takes in x (the features T1 to E), the action pair u = (u1,u2) (u1=1 when 
# RTI on, u2 = 1 when PI on; 0 otherwise for both), and the time horizon t, and 
# computes the transition from x_0 to x_t having taken action u1 u2. Returns 
# the new features x_t (T1, T2... V, E)
def f(x, u, t=5, dt = 0.0005, log=True):
    #print(x)
    if log: 
        x = np.array([10**y for y in x])
    T1, T2, Ts1, Ts2, V, E = x
    #print(T1, T2, Ts1, Ts2, V, E)
    try:
        u1, u2 = u
    except: 
        u = convert(u)
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
        
    x_new = (T1, T2, Ts1, Ts2, V, E)
    
    if log:
        return np.log10(x_new)
    
    return x_new

# c_a() takes in a tuple of features x, and a tuple of actions u = (u1, u2),
# and returns the action-dependent cost of the current state-action pair
def c_a(x, u):
    
    T1, T2, Ts1, Ts2, V, E = x
    try:
        u1, u2 = u
    except: 
        u = convert(u)
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
# x_df is a dataframe of the form ['x_t', 'u', 'x_t2', 'c', 'a0', 'a1', 'a2', 'a3'] 
# for each one-step transition returns the last iteration QK, and a function \
# policy that takes a state and outputs the optimal action
def fitted_Q(K, x_df, gamma):
    
    x_df = x_df.copy(deep=True)
    # initialize storage and actions
    #Qs = []
    
    # create the first Q1 function
    class Q:
        def predict(self, array): 
            x = array[0][:6]
            u = array[0][6:]
            return c_a(x, u)
    Q_new = Q()
    #Qs.append(Q_new)
    
    # create X using x_t and u
    # select (x_t, u) pair as training
    X = x_df['x_t'].apply(pd.Series).merge(x_df['u'].apply(pd.Series), \
                left_index = True, right_index = True) 
    print('New training features', flush=True)
    print(X, flush=True)
    
    
    bar = tqdm(range(K))
    #bar = range(K)
    #creating new Qk functions
    for i in bar:
        # create y using Qk-1 and x_t2
        # non-DP
        y = x_df.apply(lambda x: x.c + gamma*min([Q_new.predict([f]) \
                                for f in [np.array(x.a0), np.array(x.a1), \
                                          np.array(x.a2), np.array(x.a3)]]), axis=1)
        
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
        #print(y)
        y = np.array(y)
        # train the actual Regression function as Qk
        #regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
        #regr = LinearRegression().fit(X, y)
        #regr = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
        regr = ExtraTreesRegressor(n_estimators=50).fit(X,y.ravel())
        #Qs.append(regr)
        Q_new = regr
        #print('memo size', len(memo), 'used', mu, flush=True)
        
    
    #QK = Qs[-1]
    QK = Q_new
    
    p = policy()
    p.fit(QK)
        
    return QK, p#, Qs


# predict() takes the current Q 
def predict(Q, f, memo):
    if f in memo: 
        return memo[f]
    else:
        q = Q.predict([f])
        memo[f] = q
        return q


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
        p,
        x_init= (163573, 5, 11945, 46, 63919, 24)): # a trained policy model
    # starting in non-healthy steady state
    
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
            xt = f(x, u, 5, log=False) # simulating 5 day change
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
                  eps, # if cost == 'dist', thresh is threshold around healthy & unhealthy points
                  id_start = 0,
                  p=None): 
    
    # starting in non-healthy steady state
    unhealthy = (163573, 5, 11945, 46, 63919, 24)
    healthy = (967839, 621, 76, 6, 415, 353108)
    
    
    transitions = []
    # creating patient transitions
    for i in range(id_start, id_start+N):
        ID = i
        x = [0, 0, 0, 0, 0, 0] # TODO: update for randomness in starting state here as well
        x[0] = np.random.randint(10**5, 10**5.3)
        x[1] = np.random.randint(10**0.5, 10**1.5)
        x[2] = np.random.randint(10**3, 10**4.5)
        x[3] = np.random.randint(10**1, 10**1.8)
        x[4] = np.random.randint(10**4, 10**5)
        x[5] = np.random.randint(10**1, 10**2)
        for t in range(T):
            # smaller than r means takes random action, else optimal policy
            TIME = t
            if random.random() <= r: 
                u = (np.random.randint(0, 2), np.random.randint(0, 2))
            else:
                u = p.get_action(x)
            
            xt = f(x, u, 5, log=False) # simulating 5 day change
            
            if cost == 'dist': 
                if distance(x, eps):
                    c = 1
                else: 
                    c = 0
            if cost == 'c_a':
                c = c_a(x, u)
            if cost == 'c_r':
                c = c_r(x)
            a = convert(u)
            x_log = np.log10(np.array(x))
            transitions.append([ID, TIME, x_log, a, c])
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
    try: 
        len(u)
    except:
        u = int(u)
    if type(u) == int: 
        return Us[u]
    else:
        return Us.index(u)
    
# J() calculates the converging cost given a policy and N from a trained fitted Q
def J(x, policy, T, gamma):
    total = 0
    a = policy.get_action(x)
    for i in range(T):
        total += (gamma**i)*c_a(x, a)
        x = f(x, a, 5, log=False)
        a = policy.get_action(x)
    return total


# trajectory() takes two features (from 0 to 5), and a list of actions (0 to 3),
# and plots the trajectories (on a log10 scale) of the two features given 
# these actions. Returns the lists of points xs, and ys
def trajectory(f1, f2, actions):
    x = (163573, 5, 11945, 46, 63919, 24)
    healthy = (967839, 621, 76, 6, 415, 353108)
    
    xs = [np.log10(x[f1])]
    ys = [np.log10(x[f2])]
    for a in actions:
        xt = f(x, convert(a), 5, log=False)
        xs.append(np.log10(xt[f1]))
        ys.append(np.log10(xt[f2]))
        x = xt
    #print(xs[:10])
    #print(ys[:10])
    print('last:', xs[-1], ys[-1])
    print('healthy:', np.log10(healthy[f1]), np.log10(healthy[f2]))
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
    ax.set_xlabel('FEATURE_%i' %f1)
    ax.set_ylabel('FEATURE_%i' %f2)
    plt.show()
    return xs, ys
    

# distance() takes a state x and a percentage error eps, and 
# returns True if log10(x) is within eps of healthy state for all 6 features
def distance(x, eps):
    healthy = np.array((967839, 621, 76, 6, 415, 353108))
    healthy_log = np.log10(healthy)
    x = np.log10(np.array(x))
    h = True
    for i in range(6): 
        if abs(healthy_log[i] - x[i]) > healthy_log[i]*eps:
            h = False
            break
    return h


# distance_df() takes a dataframe and and epsilon, and returns a dataframe
# of True/Falses as well as the count of each
def distance_df(df, eps):
    '''
    dfx = df['x_t']
    count = 0 
    for x in dfx:
        y = distance(x, eps)
        dfx['within '+str(eps)] = y
        if y:
            count += 1
    '''
    df[str(eps)] = df.apply(lambda x: distance(x.x_t, eps),axis=1)
    count = df.loc[df[str(eps)]==True]['x_t'].count()
    return count, df
        
# plot_counts() takes a dataframe and a list of epsilons to try, and plots
# epsilons with the number of datapoints from df that are withing epsilon
def plot_counts(df, eps_list):
    counts = []
    for ep in eps_list:
        count, df = distance_df(df, ep)
        counts.append(count)
    fig, ax = plt.subplots()
    ax.plot(eps_list, counts)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('# of Datapoints')
    ax.title.set_text('Number of datapoints within various Epsilons of Healthy State')
    plt.show()
    return counts
    
# min_distance() takes a dataframe of that has an x_t column, and returns the 
# minimum distance (percentage) of the closest point to healthy state, as well
# as the index of this occurance
def min_distance(df):
    #healthy = (967839, 621, 76, 6, 415, 353108)
    dfx = df['x_t']
    
    min_d = float('inf')
    loc = None
    for i in range(len(dfx)): 
        d = get_dist(dfx[i])
        if d < min_d:
            min_d = d
            loc = i
            
    return min_d, loc
        
# get_dist() gets the distance of a datapoint from the healthy state
def get_dist(x):
    healthy = (967839, 621, 76, 6, 415, 353108)
    max_d = 0
    for i in range(6):
        d = abs(healthy[i] - x[i])/healthy[i]
        if d > max_d:
            max_d = d
    return max_d


# create_opt_samples() takes a policy and int opt_s deciding how many paths of optimal
# policy to generate, generates according to policy for first 80 steps, 
# then takes (0, 0) for next steps. Takes another int rand_s that determines
# number of semi-random paths to generate, following 0.15 randomness and policy 
# p the rest of the time. Returns a dataframe of this full path, in form suitable
# to run the algorithm. 
def create_opt_samples(p, opt_s, rand_s, eps, t_max):
    
    df_final = pd.DataFrame()
    # first create the optimal samples
    for i in range(opt_s):
        # first 80 steps according to policy
        df_opt = createSamples(1, 80, 0, 'dist', eps, i, p)
        x = df_opt.iloc[-1, 2:8]
        x = 10**x
        
        # create next 120 steps according to (0, 0) action
        transitions = []
        ID = i
        for t in range(80, t_max):
            TIME = t
            u = (0, 0)
            xt = f(x, u, 5, log=False) # simulating 5 day change
            
            if distance(xt, eps):
                c = 1
            else: 
                c = 0

            a = convert(u)
            x_log = np.log10(np.array(xt))
            transitions.append([ID, TIME, x_log, a, c])
            x = xt
    
        df = pd.DataFrame(transitions, columns=['ID', 'TIME', 'x', 'ACTION', 'RISK'])
        features = df['x'].apply(pd.Series)
        features = features.rename(columns = lambda x : 'FEATURE_' + str(x))
        
        df_new = pd.concat([df.iloc[:, :2], features, df.iloc[:, 3:]], axis=1)
        df_opt = pd.concat([df_opt, df_new], ignore_index=True)
        df_final = pd.concat([df_final, df_opt], ignore_index=True)
    
    # next create the partially random samples
    df_rand = createSamples(rand_s, t_max, 0.15, 'dist', eps, opt_s, p)
    df_final = pd.concat([df_final, df_rand], ignore_index=True)
        
    return df_final


# create_opt_path() takes a starting state x_init, generates according to policy
# for first 80 steps, then takes (0, 0) for next steps. Returns a dataframe
# of this full path
def create_opt_path(x_init, p, plot=False):
    df10a = gen(1, 80, 0, 'c_a', p, x_init)
    x = df10a['x_t'].iloc[-1]
    
    transitions = []
    for t in range(120):
        # smaller than r means takes random action, else optimal policy
        u = (0, 0)
        xt = f(x, u, 5, log=False) # simulating 5 day change
        c = c_a(x, u)
        transitions.append([x, u, xt, c])
        # update new x
        x = xt
        #print('t', t, 'x', x)
    
    df = pd.DataFrame(transitions, columns=['x_t', 'u', 'x_t2', 'c'])
    
    df_new = pd.concat([df10a, df], ignore_index=True)
    
    if plot:
        for i in range(6):
            fig, ax = plt.subplots()
            ax.plot(df_new['x_t'].apply(lambda x: np.log10(x[i])))
            plt.show()
        
    return df_new 


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
    