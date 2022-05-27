# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:52:42 2022

@author: Isabel Thomas isabel.thomas@unige.ch
"""
##import modules 
import pandas as pd
import os
import numpy as np
import sklearn
import random

#%% 1. set up file path
ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"

#%% 2. read in pre-prepared data
# where each instance has been assigned
# to a cell as descibed in methods

#data for 1985 contains data from 2 landsat images
#data for 1997 contains data from 5 landsat images

df85 = pd.read_csv(os.path.join(ml_dir, "full_df/points_85.csv"))
df97 = pd.read_csv(os.path.join(ml_dir, "full_df/points_97.csv"))

#%% 3. cropping the dataset to a smaller area, to make it more manageable to run
# on laptop (x coordinate less than 2541100)

df85 = df85[df85['E']<2541100]
df97 = df97[df97['E']<2541100]

#%% 4. clean up data

df85 = df85[['B1', 'B2', 'B3', 'B4','B5', 'LC85_6','index_right']]
df97 = df97[['B1', 'B2', 'B3', 'B4','B5', 'LC97_6','index_right']]
df85 = df85.rename(columns={"LC85_6": "LC"})
df97 = df97.rename(columns={"LC97_6": "LC"})

df = pd.concat([df85,df97])

#%% 5. separate out test set 
#get unique values of cell identifier
cells = df['index_right'].drop_duplicates()
#set random seed
random.seed(41)
#select 20 cells
cells_test = random.sample(list(cells), 20)
#separate test and training/validation based on these 20 cells
df_test = df[df['index_right'].isin(cells_test)]
df_tv = df[-df['index_right'].isin(cells_test)]

#%% 6. Split dataset into X (inputs) and y (labels)
#training/validation
X_tv = df_tv[['B1', 'B2', 'B3', 'B4', 'B5']]
X_tv = X_tv.to_numpy()
y_tv = df_tv[['LC']]
y_tv = y_tv.values.ravel()
#test
X_test = df_test[['B1', 'B2', 'B3', 'B4', 'B5']]
X_test = X_test.to_numpy()
y_test = df_test[['LC']]
y_test = y_test.values.ravel()

#%% 7. Print class distributions
print(pd.value_counts(y_tv))
print(pd.value_counts(y_test))

#%% 8. Define training/validation splits

# Grouped K-fold cross validation is used to maintain the predefined group separation to 
# avoid overestimation of model due to spatial autocorrelation from selecting 
# training & validation data from neighbouring datapoints. 

from sklearn.model_selection import GroupKFold

#unique identifier for grid cell ('index_right' as grouping variable)
groups = df_tv['index_right'].values
#initiate GroupKFold with 7 folds
group_kfold = GroupKFold(n_splits=7) 

# Generator for the train/test indices
fold_kfold = group_kfold.split(X_tv, y_tv, groups)  

# Create a nested list of train and test indices for each fold
train_indices, val_indices = [list(trainval) for trainval in zip(*fold_kfold)]

fold_cv = [*zip(train_indices, val_indices)]

#%% Optional: check train/validation average class distribution
#y_tv = df_tv[['LC']]
#y_train_vals = ((pd.value_counts(y_tv.iloc[fold_cv[0][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[1][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[3][0]].values.ravel()))/7)
#y_valid_vals = ((pd.value_counts(y_tv.iloc[fold_cv[0][1]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[1][1]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][1]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[3][1]].values.ravel()))/7)

#print(y_train_vals)
#print(y_valid_vals)
#y_tv = y_tv.values.ravel()

#%% 9. Define parameter grid for hyperparameter search
# n_estimators=400 originally included, but not recommended for laptop use...
param_grid = {'n_estimators': [50, 100, 150, 200],
               'max_depth': [10,25,50],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,10,25]}
#%% 10. Initiate RF classifer, train model whilst performing hyperparameter search
### currently takes ~6 hours ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
import time

#initiate timer for whole classification
t0 = time.time()

#initiate classifier
rfc = RandomForestClassifier(random_state=42)

#run with accuracy as performance metric
#grid_search = HalvingGridSearchCV(rfc, param_grid, cv=fold_cv, verbose=3)

#to run with f1 score as performance metric (f1_weighted as multiclass classification)
grid_search = HalvingGridSearchCV(rfc, param_grid, cv=fold_cv, scoring='f1_weighted', verbose=3)

grid_search.fit(X_tv, y_tv)

#print best parameters
print("The best parameters are %s with a score of %0.2f"
    % (grid_search.best_params_, grid_search.best_score_))

#calculate timer
t1 = time.time()
print(f"{(t1 - t0):.2f}s elapsed")  

#%% 9. Save RF results
results = grid_search.cv_results_
df = pd.DataFrame(results) 
df.to_csv('C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/grid_search_f1-2705.csv')

#%% 10. Predict test set values
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 

y_pred = grid_search.predict(X_test)

# Performance metrics on test set
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="weighted")

print(f'The accuracy of the model on the test set is {test_acc:.1%}')
print(f'The f1 score of the model on the test set is {test_f1:.1%}')

#%% 11. Return class-based metrics
from sklearn.metrics import confusion_matrix, classification_report

#(to load in pre-saved results)
#test_0409 = pd.read_csv(os.path.join(ml_dir, "results_for_test_2004-2009_2505.csv"))
#y_pred = test_0409['preds']

y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')

print(classification_report(y_test, y_pred))

#confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)
#precision
df_conf_norm_a = df_confusion / df_confusion.sum(axis=1)
#recall
df_conf_norm_b = df_confusion / df_confusion.sum(axis=0)
matrix = confusion_matrix(y_test, y_pred, normalize='pred')

#%% 12. save predictions and probabilities

all_probs = np.max(grid_search.predict_proba(X_test), axis=1)
df_test['prob'] = all_probs
df_test['predicted'] = y_pred

df_test.to_csv(os.path.join(ml_dir, "results_for_test_2004-2009_2505.csv"))
#%% 13. Calculate kappa score
from sklearn.metrics import cohen_kappa_score
kscore = cohen_kappa_score(y_test, y_pred)
print(kscore)

#%% 14. Second test set (2004 Landsat image data)
test_0409 = pd.read_csv(os.path.join(ml_dir, "data/test_2004-2009.csv"))
X_test2 = df_test[['B1', 'B2', 'B3', 'B4', 'B5']]
X_test2 = X_test2.to_numpy()
y_test2 = df_test[['LC09R_6']]
y_test2 = y_test2.values.ravel()

test_preds = grid_search.predict(X_test2)

test_acc = accuracy_score(y_test2, test_preds)
test_f1 = f1_score(y_test2, test_preds, average="weighted")

print(f'The accuracy of the model on the test set is {test_acc:.1%}')
print(f'The f1 score of the model on the test set is {test_f1:.1%}')
#%% 15. Calculate class-based performance metrics
from sklearn.metrics import confusion_matrix
y_actu = pd.Series(y_test2, name='Actual')
y_pred = pd.Series(test_preds, name='Predicted')

#confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)
#precision
df_conf_norm_a = df_confusion / df_confusion.sum(axis=1)
#recall
df_conf_norm_b = df_confusion / df_confusion.sum(axis=0)
matrix = confusion_matrix(y_test, y_pred, normalize='pred')
#%% 16. Run Multinomial Logistic Regression model with hyperparameter search
#parameter grid
param_grid = {'solver': ['saga', 'sag'],
               'C': [100, 10, 1.0, 0.1, 0.01]}

from sklearn.linear_model import LogisticRegression
import time
t0 = time.time()

log_mod = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=4000)

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV 

grid_search = HalvingGridSearchCV(log_mod, param_grid, cv=fold_cv, scoring='f1_weighted', verbose=3)
#grid_search = HalvingGridSearchCV(log_mod, param_grid, cv=fold_cv, scoring='f1_weighted', verbose=3)
grid_search.fit(X_tv, y_tv)
t1 = time.time()
results = grid_search.cv_results_
df = pd.DataFrame(results) 
df.to_csv('C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/grid_search_lm2405_f1.csv')
#%%  MLR predictions on test set
test_preds1 = grid_search.predict(X_test)

test_acc = accuracy_score(y_test, test_preds1)
test_f1 = f1_score(y_test, test_preds1, average="weighted")

print(f'The accuracy of the model on the test set (1) is {test_acc:.1%}')
print(f'The f1 score of the model on the test set (1) is {test_f1:.1%}')

test_preds2 = grid_search.predict(X_test2)

test_acc = accuracy_score(y_test2, test_preds2)
test_f1 = f1_score(y_test2, test_preds2, average="weighted")

print(f'The accuracy of the model on the test set (2) is {test_acc:.1%}')
print(f'The f1 score of the model on the test set (2) is {test_f1:.1%}')