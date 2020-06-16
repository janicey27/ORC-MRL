#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:37:12 2020

@author: janiceyang
"""

# f() simulates the transitions for actions 0 and 1. It takes a state x which
# is a (r, theta) pair, then performs transition for actions inwards by 1 unit
# (action 0), or outwards by 1 unit (action 1).
# returns the new x tuple
def f(x, a):
    # if x in the reward sink already, no change
    if x[0] <= 0:
        return x
    # move those who just got to reward to the sink (r <= 0)
    elif x[0] <= 1: 
        return x[0]-1, x[1]
    else:
        if a == 0:
            return x[0]-1, x[1]
        elif a == 1:
            return x[0]+1, x[1]
        else:
            print('Action not seen')
            return 
        
# reward() takes a state x and returns the corresponding reward of reaching
# this state. Reward is only 1 when 0 < r <= 1
def reward(x):
    if x[0] > 0 and x[0] <= 1:
        return 1
    else: 
        return 0
    
    
# generate dataset
        