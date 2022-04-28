# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:08:11 2022

@author: isabe
"""

import pandas as pd
import xarray as xr
df = pd.read_csv("C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/test_datav2.csv")
#df = pd.read_csv("C:/Users/isabe/Downloads/sub.csv")
df = df[['new_x', 'new_y', 'LC09R_6']]
#df = df[['E', 'N', 'LC09R_6']]
#df = df[df['E'] < 2560000]
#df = df[df['E'] > 2540000]
#df = df[df['N'] < 1160000]
#df = df[df['N'] > 1140000]
df = df[df['new_x'] < 800000]
df = df[df['new_x'] > 785000]
df = df[df['new_y'] < 5200000]
df = df[df['new_y'] > 5185000]
print(df.shape)
#%% OLD CODE
def np_to_xr(array_name, array, N, E):
    #building the xarrray
    da = xr.DataArray(data = array, # Data to be stored
                  
                  #set the name of dimensions for the dataArray 
                  dims = ['y_coord','x_coord'],
                  
                  #Set the dictionary pointing the name dimensions to np arrays 
                  coords = {'y_coord':N,
                            'x_coord':E},
                      
                  name=array_name)
    return da
#%% OLD CODE
E = df['E'].to_numpy()
N = df['N'].to_numpy()
cat = df['LC09R_6'].to_numpy
#%% OLD CODE
ds = np_to_xr("LC", cat, N, E)
#%%
#df = df[['new_x', 'new_y', 'LC09R_6']]
#print(df.head)
df = df.set_index(['new_x', 'new_y'])
#df = df.set_index(['E', 'N'])
test = df.to_xarray()
print(test)