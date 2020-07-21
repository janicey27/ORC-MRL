#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:10:34 2020

@author: janiceyang
"""
#################################################################
# Loading Modules
from HIV_functions import gen, fitted_Q

import pandas as pd
import pickle
from datetime import datetime
import ast
#################################################################
# Running the experiment
d1 = datetime.now()
print('Starting Time:', d1, flush=True)

#policies = []
# creating first 6000, completely random
#df_all = gen(30, 200, 1, 'c_a', None)
#Q, p = fitted_Q(2, df_all, 0.98) #change 10 to 400
#policies.append(p)
#Q_opt = Q
#all_Qs = Qs

# opening dataset from previous round
df_all = pd.read_csv('set9.csv')
c = df_all['c']

try:
    # just in case ID became an unnamed column
    df_all.drop(columns = ['Unnamed: 0', 'c'], inplace=True)
except: 
    df_all.drop(columns = ['c'], inplace=True)

# changing strings to tuples again
df_all = df_all.applymap(ast.literal_eval)
df_all.insert(2, 'c', c)
#print(df_all)
p = pickle.load(open('p9.sav', 'rb'))


# creating new samples via 15% random, 85% optimal policy from ALL 
# prior samples
for i in range(1):
    print('starting set ', i)
    df2 = gen(30, 200, 0.15, 'c_a', p)
    df_all = pd.concat([df_all, df2], ignore_index=True, sort=False)

    Q, p = fitted_Q(400, df_all, 0.98) #change 10 to 400
    #policies.append(p)
    Q_opt = Q
    #all_Qs = Qs
    
    # displaying the optimal policies here
    df_test = gen(1, 60, 0, 'c_a', p)
    print(df_test['u'], flush=True)
    print(df_test[['x_t', 'c']], flush=True)


# exporting dataset
df_all.to_csv('set10.csv')
# saving the final policy and last set of Q's
filename = 'p10.sav'
pickle.dump(p, open(filename, 'wb'))

#with open("Qs_2.sav", "wb") as file:
    #for q in all_Qs[1:]:
        #pickle.dump(q, file)

d2 = datetime.now()
print('Ending Time:', d2, flush=True)
print('Time elapsed:', d2-d1, flush=True)


#################################################################
# Script for accessing final policy and final Q's

# =============================================================================
# policy = pickle.load(open('final_policy.sav', 'rb'))
# 
# models = []
# with open("Qs.sav", "rb") as f:
#     while True:
#         try:
#             models.append(pickle.load(f))
#         except EOFError:
#             break
# =============================================================================
