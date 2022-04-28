# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:21:38 2022

@author: isabe
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:03:33 2022

@author: isabe
"""
import pandas as pd
import os
import rasterio as rio
import re
import numpy as np

#%% set up file paths

ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"
point_data = 'LC09_point_l5clip.csv'
rast_stack_name = "data/2004_2009/stackd/stack_20040907.tif"
tif_folder = "data/2004_2009/test_folder/"
#%% read in point data 

df = pd.read_csv(os.path.join(ml_dir, point_data))
print(df.shape)
df=df[['new_x', 'new_y', 'LC_6']]
print(df.shape)        

#%%

r = re.compile(".*B.*\.TIF$")
dir_list =  os.listdir(os.path.join(ml_dir, "data/2004_2009/test_folder/"))
grid_list = list(filter(r.match,dir_list))
grid_list = grid_list[0:5]

#%%

df.index = range(len(df))
coords = [(x,y) for x, y in zip(df.new_x, df.new_y)]

src1 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[0]))
df['B1'] = [x[0] for x in src1.sample(coords)]
src2 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[1]))
df['B2'] = [x[0] for x in src2.sample(coords)]
src3 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[2]))
df['B3'] = [x[0] for x in src3.sample(coords)]
src4 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[3]))
df['B4'] = [x[0] for x in src4.sample(coords)]
src5 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[4]))
df['B5'] = [x[0] for x in src5.sample(coords)]

#%%

X_sub = []
y_sub = []

from sklearn.utils import shuffle
for subclass in [10,20,30,40,50,60]:
    sub = df[df['LC_6']==subclass]
    sub = shuffle(sub)
    sub = sub[0:14000]
    X_subt = sub[['B1', 'B2', 'B3', 'B4','B5']]
    y_subt = sub[['LC_6']]
    X_sub.append(X_subt)
    y_sub.append(y_subt)
    
X_sub = pd.concat(X_sub)
y_sub = pd.concat(y_sub)

print(y_sub.value_counts())

#%% Create balanced train, test and validation set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
X = df[['B1', 'B2', 'B3', 'B4','B5']]
X = X.to_numpy()
#X = X_sub.to_numpy()
y_pd = df[['LC_6']]
#y_pd = y_sub
y = y_pd.values.ravel()


sss = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
sss.get_n_splits(X, y)
print(sss)
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("test_value_counts:", pd.value_counts(y[test_index]))

print(pd.value_counts(y_train))
print(pd.value_counts(y_test))

#%%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train) 

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 

rfc_preds = rfc.predict(X_test)

rfc_acc = accuracy_score(y_test, rfc_preds)
print(f'The accuracy of the model is {rfc_acc:.1%}')

rfc_f1 = f1_score(y_test, rfc_preds, average="weighted")
print(f'The f1 score of the model is {rfc_f1:.1%}')

#%%
param_grid = {'n_estimators': [50, 100, 150, 200, 400],
               'max_depth': [10,25,50],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,10,25,50]}

#%%
import sys
sys.stdout = open(os.path.join(ml_dir,'log2.txt'), 'w')

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sys
sys.stdout = open(os.path.join(ml_dir,'log_2704.txt'), 'w')

rfc = RandomForestClassifier()

cv = StratifiedShuffleSplit(n_splits=4, test_size=0.2, random_state=42)
cv.get_n_splits(X, y)

#grid_search = GridSearchCV(rfc, param_grid, cv=cv, verbose=3)
#grid_search.fit(X, y)

grid_search = HalvingGridSearchCV(rfc, param_grid, cv=cv, verbose=3)
grid_search.fit(X, y)
sys.stdout.close()
#%%
print(    "The best parameters are %s with a score of %0.2f"
    % (grid_search.best_params_, grid_search.best_score_)
)