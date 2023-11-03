# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:01:26 2022

@author: sizhean
"""


import pandas as pd
import numpy as np
import os
import argparse
import warnings

# %%
def main(subject):

    warnings.filterwarnings("ignore")
    in_path = '../aligned_data/radar/singleframe/'
    df_radar = pd.read_csv(in_path + subject + '.csv')
    out_path = '../features/radar/'
    
    # #drop unusual frames
    # df_radar = df_radar.drop(df_radar[(df_radar.X < -1) | (df_radar.X > 1)].index)
    # df_radar = df_radar.drop(df_radar[(df_radar.Y < 0) | (df_radar.Y > 2)].index)
    # df_radar = df_radar.drop(df_radar[(df_radar.Z < 0) | (df_radar.Z > 1)].index)
    
    # convert to array get r infomation
    df_radar_array = np.asarray(df_radar)
    
    
    #normalize the intensity
    mu = np.mean(df_radar_array[:,6], axis = 0)
    sigma = np.std(df_radar_array[:,6], axis = 0)
    normed_intensity = (df_radar_array[:,6] - mu)/sigma
    df_radar['Intensity_normed'] = normed_intensity
    all_frames = np.unique(df_radar_array[:,8])
    
    # 8*8 = 64 points
    # 14*14 = 196 points
    n_dim = 14
    
    
    
    
    xyzdi_featuremap_all = np.zeros((1,n_dim,n_dim,5))
    
    for i in range(len(all_frames)):
        x_coor = np.zeros((n_dim,n_dim))
        y_coor = np.zeros((n_dim,n_dim))
        z_coor = np.zeros((n_dim,n_dim))
        doppler = np.zeros((n_dim,n_dim))
        intensity = np.zeros((n_dim,n_dim))
        
        # 0 -> all_frames[0]
        relative_frame = i + all_frames[0]
        
        if i == 0:
            df_entry = df_radar[df_radar['Camera Frame'] == (relative_frame)]
            df_entry = df_entry.append(df_radar[df_radar['Camera Frame'] == (relative_frame+1)])
            df_entry = df_entry.append(df_radar[df_radar['Camera Frame'] == (relative_frame+2)])
        elif i ==len(all_frames) - 1:
            df_entry = df_radar[df_radar['Camera Frame'] == (relative_frame-2)]
            df_entry = df_entry.append(df_radar[df_radar['Camera Frame'] == (relative_frame-1)])
            df_entry = df_entry.append(df_radar[df_radar['Camera Frame'] == (relative_frame)])
        else:
            df_entry = df_radar[df_radar['Camera Frame'] == (relative_frame-1)]
            df_entry = df_entry.append(df_radar[df_radar['Camera Frame'] == (relative_frame)])
            df_entry = df_entry.append(df_radar[df_radar['Camera Frame'] == (relative_frame+1)])
            
        df_entry = df_entry.sort_values(by=['X', 'Y', "Z"])
        for j in range(len(df_entry)):
            row = int(j/n_dim)
            col = int(j%n_dim)
            x_coor[row][col] = df_entry.iloc[j]['X']
            y_coor[row][col] = df_entry.iloc[j]['Y']
            z_coor[row][col] = df_entry.iloc[j]['Z']
            doppler[row][col] = df_entry.iloc[j]['Doppler']
            intensity[row][col] = df_entry.iloc[j]['Intensity_normed']
    
        xyzdi_featuremap = np.dstack((x_coor, y_coor, z_coor, doppler, intensity)).reshape(1,n_dim,n_dim,5)
        xyzdi_featuremap_all = np.concatenate((xyzdi_featuremap_all, xyzdi_featuremap))
    
    xyzdi_featuremap_all = xyzdi_featuremap_all[1:len(xyzdi_featuremap_all)]
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    
    np.save(out_path + subject + "_featuremap.npy", xyzdi_featuremap_all)

# %%
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-subj', '--subject', type=str, default = 'subject1',
           help='get featuremap for which subject')
    main(**vars(p.parse_args()))