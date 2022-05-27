# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:33:35 2022
Script to extract data from Landsat-5 images at (reprojected) points 
for land cover data
@author: isabe
"""

import pandas as pd
import os
import rasterio as rio
import re

#%% set up file paths

ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"
point_data = "full_df/proj_points_folds.csv"
tif_folder = "data/2004_2009/test_folder/"
extract_dir = os.path.join(ml_dir, "data/1992_1997/extracted")
dir_list =  os.listdir(extract_dir)

#%% read in point data 

df = pd.read_csv(os.path.join(ml_dir, point_data))
print(df.shape)
df=df[['new_x', 'new_y', 'LC85_6', 'LC97_6', 'LC09R_6', 'LC18_6', 'folds']]     
print(df.shape)

#%%

df_all=[]

for d in dir_list:
    print("processing_", d)
    files =  os.listdir(os.path.join(extract_dir, d))
    #regex for L5 band X tif files 
    r = re.compile(".*B.*\.TIF$")
    grid_list = list(filter(r.match,files))
    #select only the first 5 bands
    grid_list = grid_list[0:5]
    print(grid_list)
    
    #load point data
    df = pd.read_csv(os.path.join(ml_dir, point_data))
    df.index = range(len(df))
    #extract coordinates
    coords = [(x,y) for x, y in zip(df.new_x, df.new_y)]
    
    #each band is saved as a separate tif, load tif and extract point data
    src1 = rio.open(os.path.join(extract_dir, d, grid_list[0]))
    df['B1'] = [x[0] for x in src1.sample(coords)]
    print("B1")
    src2 = rio.open(os.path.join(extract_dir, d, grid_list[1]))
    df['B2'] = [x[0] for x in src2.sample(coords)]
    print("B2")
    src3 = rio.open(os.path.join(extract_dir, d, grid_list[2]))
    df['B3'] = [x[0] for x in src3.sample(coords)]
    print("B3")
    src4 = rio.open(os.path.join(extract_dir, d, grid_list[3]))
    df['B4'] = [x[0] for x in src4.sample(coords)]
    print("B4")
    src5 = rio.open(os.path.join(extract_dir, d, grid_list[4]))
    df['B5'] = [x[0] for x in src5.sample(coords)]
    print("B5")
    #add column with L5 image indentifier
    df['srce'] = re.search('LT05_L2SP_\d{6}_(.+?)_', d).group(1)
    
    df_all.append(df)

df_all = pd.concat(df_all)

df_all.to_csv('C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/full_df/points_97.csv', mode='a')