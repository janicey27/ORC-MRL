# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""

import pandas as pd
import matplotlib.pyplot as plt

#maps each OG_ClUSTER to a cluster --> We map a cluster to the OG_CLUSTER most 
#present in it
def get_predictions(df_new):
    df0 = df_new.groupby(['CLUSTER', 'OG_CLUSTER'])['ACTION'].count()
    df0 = df0.groupby('CLUSTER').idxmax()
    df2 = pd.DataFrame()
    df2['OG_CLUSTER'] = df0.apply(lambda x: x[1])
    return df2



#Returns the global training accuracy and a df of training accuracy per OG_CLUSTER
def training_accuracy(df_new):
    clusters = get_predictions(df_new)
    #First term is what the algo predicts for each training data points, sets 
    #term is what is the truth
    accuracy = clusters.loc[df_new['CLUSTER']].reset_index()['OG_CLUSTER'] == df_new['OG_CLUSTER']
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame('Accuracy')
    accuracy_df['OG_CLUSTER'] = df_new['OG_CLUSTER']
    accuracy_df = accuracy_df.groupby('OG_CLUSTER').mean()
    return (tr_accuracy,accuracy_df)

# Returns the purity of each cluster
def Purity(df):
    su = pd.DataFrame(df.groupby(['CLUSTER'])['OG_CLUSTER']
    .value_counts(normalize=True)).reset_index(level=0)
    su.columns= ['CLUSTER','Purity']
    return su.groupby('CLUSTER')['Purity'].max()



def plot_features(df):
    x=  list(df['FEATURE_1'])
    y=  list(df['FEATURE_2'])
    plt.scatter(x, y)
    plt.show()