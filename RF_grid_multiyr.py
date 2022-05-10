# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:52:42 2022

@author: Isabel Thomas isabel.thomas@unige.ch
"""
import pandas as pd
import os
import rasterio as rio
import re
import numpy as np
import sklearn

#%% set up file paths

ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"
#point_data = "points_w_folds.csv"
#rast_stack_name = "data/2004_2009/stackd/stack_20040907.tif"
tif_folder = "data/2004_2009/test_folder/"

#%% read in pre-prepared train/validation data, where each instance has been assigned
#to a fold

#data for 1985 contains data from 2 landsat images
#data for 1997 contains data from 5 landst images

df85 = pd.read_csv(os.path.join(ml_dir, "points_w_folds85.csv"))
df97 = pd.read_csv(os.path.join(ml_dir, "points_w_folds97.csv"))

#%%
#clean up data
df85 = df85[['B1', 'B2', 'B3', 'B4','B5', 'LC85_6','folds']]
df97 = df97[['B1', 'B2', 'B3', 'B4','B5', 'LC97_6','folds']]
df85 = df85.rename(columns={"LC85_6": "LC"})
df97 = df97.rename(columns={"LC97_6": "LC"})

df = pd.concat([df85,df97])
#split according to grid
df_train = df[df['folds'].isin([1,2,3,4,5,6,7])]
df_valid = df[df['folds'].isin([8,9,10])]
#%% split datasets into X (inputs) and y (labels)

X_train = df_train[['B1', 'B2', 'B3', 'B4','B5']]
X_train = X_train.to_numpy()
y_train = df_train[['LC']]
y_train = y_train.values.ravel()

X_valid = df_valid[['B1', 'B2', 'B3', 'B4','B5']]
X_valid = X_valid.to_numpy()
y_valid = df_valid[['LC']]
y_valid = y_valid.values.ravel()

print(pd.value_counts(y_train))
print(pd.value_counts(y_valid))

#%% define split for cross validation to keep current distribution for train and valid sets.
#https://stackoverflow.com/questions/31948879/using-explicit-predefined-validation-set-for-grid-search-with-sklearn

from sklearn.model_selection import PredefinedSplit

split_index = [-1]*len(X_train) + [0]*len(X_valid)
X = np.concatenate((X_train, X_valid), axis=0)
y = np.concatenate((y_train, y_valid), axis=0)

pds = PredefinedSplit(test_fold = split_index)

pds.get_n_splits()
# OUTPUT: 1
#%% Parameter grid for hyperparameter search

param_grid = {'n_estimators': [50, 100, 150, 200, 400],
               'max_depth': [10,25,50],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,10,25]}

#%% Initiate RF classifer and perform hyperparameter search
### currently takes ~11 hours ###
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV

grid_search = HalvingGridSearchCV(rfc, param_grid, cv=pds, verbose=3, n_jobs=-1)
grid_search.fit(X, y)

#%%
print("The best parameters are %s with a score of %0.2f"
    % (grid_search.best_params_, grid_search.best_score_))

#parameters: max_depth=25, min_samples_leaf=10, min_samples_split=10, n_estimators=400;, score=(train=0.701, test=0.679) total time=57.7min

#%% Initiate and train classifier based on results of grid search
#takes ~18 mins
import time
from sklearn.ensemble import RandomForestClassifier
t0 = time.time()

rfc = RandomForestClassifier(max_depth= 25, min_samples_leaf = 10, 
                             min_samples_split = 10, n_estimators = 200,
                             class_weight="balanced")
rfc.fit(X_train, y_train) 
t1 = time.time()
print(f"{(t1 - t0):.2f}s elapsed")  

#%% Procecss test data
# This code extracts spectral data for each of the 5 bands at each point

tif_folder = "data/2004_2009/test_folder/"

r = re.compile(".*B.*\.TIF$")
dir_list =  os.listdir(os.path.join(ml_dir, "data/2004_2009/test_folder/"))
grid_list = list(filter(r.match,dir_list))
grid_list = grid_list[0:5]

#load point data
df_test = pd.read_csv(os.path.join(ml_dir, "points_w_folds.csv"))
#extract coordinates
df_test.index = range(len(df_test))
coords = [(x,y) for x, y in zip(df_test.new_x, df_test.new_y)]

#each band is saved as a separate tif, load tif and extract point data
src1 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[0]))
df_test['B1'] = [x[0] for x in src1.sample(coords)]
src2 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[1]))
df_test['B2'] = [x[0] for x in src2.sample(coords)]
src3 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[2]))
df_test['B3'] = [x[0] for x in src3.sample(coords)]
src4 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[3]))
df_test['B4'] = [x[0] for x in src4.sample(coords)]
src5 = rio.open(os.path.join(ml_dir, tif_folder, grid_list[4]))
df_test['B5'] = [x[0] for x in src5.sample(coords)]

#split into X (inputs) and y (labels)
X_test = df_test[['B1', 'B2', 'B3', 'B4','B5']]
X_test = X_test.to_numpy()
y_pd = df_test[['LC09R_6']]
y_test = y_pd.values.ravel()

#%% Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 

train_preds = rfc.predict(X_train)
valid_preds = rfc.predict(X_valid)
test_preds = rfc.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
valid_acc = accuracy_score(y_valid, valid_preds)
test_acc = accuracy_score(y_test, test_preds)

train_f1 = f1_score(y_train, train_preds, average="weighted")
valid_f1 = f1_score(y_valid, valid_preds, average="weighted")
test_f1 = f1_score(y_test, test_preds, average="weighted")

print(f'The accuracy of the model on the training set is {train_acc:.1%}')
print(f'The accuracy of the model on the validation set is {valid_acc:.1%}')
print(f'The accuracy of the model on the test set is {test_acc:.1%}')

print(f'The f1 score of the model on the training set is {train_f1:.1%}')
print(f'The f1 score of the model on the validation set is {valid_f1:.1%}')
print(f'The f1 score of the model on the test set is {test_f1:.1%}')

#%% class-based metrics

from sklearn.metrics import confusion_matrix
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(test_preds, name='Predicted')

df_confusion = pd.crosstab(y_actu, y_pred)
df_conf_norm_a = df_confusion / df_confusion.sum(axis=1)
df_conf_norm_b = df_confusion / df_confusion.sum(axis=0)
matrix = confusion_matrix(y_test, test_preds, normalize='pred')

#res = matrix.diagonal()/matrix.sum(axis=1)
#cm = confusion_matrix(y_test, rfc_preds, normalize='pred')
#df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

#%% save predictions and probabilities

all_probs = np.max(rfc.predict_proba(X_test), axis=1)
df_test['prob'] = all_probs
df_test['predicted'] = test_preds

df_test.to_csv(os.path.join(ml_dir, "results_for_test.csv"))
#%% Calculate kappa score
from sklearn.metrics import cohen_kappa_score
kscore = cohen_kappa_score(y_test, test_preds)
print(kscore)