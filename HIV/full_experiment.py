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
#################################################################
# Running the experiment

policies = []
# creating first 6000, completely random
df_all = gen(30, 200, 1, 'c_a', None)
Q, p, Qs = fitted_Q(400, df_all, 0.98) #change 10 to 400
policies.append(p)
Q_opt = Q
all_Qs = Qs


# creating new samples via 15% random, 85% optimal policy from ALL 
# prior samples
for i in range(9):
    print('starting set ', i)
    df2 = gen(30, 200, 0.15, 'c_a', p)
    df_all = pd.concat([df_all, df2], ignore_index=True)
    Q, p, Qs = fitted_Q(400, df_all, 0.98) #change 10 to 400
    policies.append(p)
    Q_opt = Q
    all_Qs = Qs
    
    # displaying the optimal policies here
    df_test = gen(1, 60, 0, 'c_a', p)
    print(df_test['u'])
    print(df_test[['x_t', 'c']])

# saving the final policy and last set of Q's
filename = 'final_policy_2.sav'
pickle.dump(policies[-1], open(filename, 'wb'))

with open("Qs_2.sav", "wb") as file:
    for q in all_Qs[1:]:
        pickle.dump(q, file)

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
