# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:42:19 2022

@author: sizhe-admin
"""

# read the output pose .cpl file and refine triangulated 3D points
#%%
import json
import numpy as np
import pickle
import cv2
from camera_calibrate import getPmat
import torch
import matplotlib.pyplot as plt
import argparse

#%%
# Todo: change to proper filenames variables for all subjects
def loadfiles(subject):
    # load 2d/3d joints point and video labels
    pose_2d_filename_l = '../result_pose_estimation/blurred_' + subject + '_color1/pose_track.cpl'
    pose_2d_filename_r = '../result_pose_estimation/blurred_' + subject + '_color0/pose_track.cpl'
    pose_3dest_filename_l = '../result_pose_estimation/blurred_' + subject + '_color1/pose_3d.cpl'
    pose_3dest_filename_r = '../result_pose_estimation/blurred_' + subject + '_color0/pose_3d.cpl'


    videolabel_name = '/disk/san49/Dataset/rawdata/videolabels/' + subject + '.json'
    
    with open(pose_2d_filename_l, 'rb') as f:  
        pose_2d_l = pickle.load(f)
    
    with open(pose_2d_filename_r, 'rb') as f:  
        pose_2d_r = pickle.load(f)    
    
    
    with open(pose_3dest_filename_l, 'rb') as f:  
        pose_3d_l = pickle.load(f)   
     
    with open(pose_3dest_filename_r, 'rb') as f:  
        pose_3d_r = pickle.load(f)     
    
    # read json file to to get Tpose index and rest
    with open(videolabel_name,'r') as load_f:
        labels = json.load(load_f)['labels']

    return pose_2d_l, pose_2d_r, pose_3d_l, pose_3d_r, labels

#%% project 3d points to 2d
def project2d(proj_mat, points3d):
    #convert it to homogenous points
    points_3d_homo = np.insert(points3d, 3, values=1, axis=1)
    #project it to pixel
    proj_2d = np.matmul(proj_mat, points_3d_homo)
    #from homogenous to 2d. [u,v,w] -> [u/w, v/w]
    proj_2d = np.divide(proj_2d[:,0:2,:], proj_2d[:,2,:][:,np.newaxis])
    return proj_2d

#%%
def get_gt_kps_3d(pose_2d_l, pose_2d_r, pose_3d_l, pose_3d_r, P1, P2):
    # load available frames for left and right
    frame_list_l = pose_3d_l['frame_list']
    frame_list_r = pose_3d_r['frame_list']
    # get intersection of l and r
    frame_list_inter = list(set(frame_list_l).intersection(frame_list_r))
    
    # load camera matrix P1, P2
    gt_kps = []
    
    # calculate triangulation result
    # this frame number start from 0 index, to total_frame - 1 index
    
    pose_2d_l_all = []
    pose_2d_r_all = []

    for frame in frame_list_inter:
        # Triangulate the left and right points.
        pose_2d_l_one = np.squeeze(pose_2d_l[frame]['kps']).T
        pose_2d_r_one = np.squeeze(pose_2d_r[frame]['kps']).T


        if len(pose_2d_r_one.shape) == 3:
            pose_2d_r_one = pose_2d_r_one[:,:,0] 


        if len(pose_2d_l_one.shape) == 3:
            pose_2d_l_one = pose_2d_l_one[:,:,0]

        pose_2d_l_all.append(pose_2d_l_one)
        pose_2d_r_all.append(pose_2d_r_one)
        
        # output: points4D – 4xN array of reconstructed points in homogeneous coordinates.
        # we need to divide the first three row by the fourth row do to the normalization to get the camera coordinates.
        
        pose_4d_one = cv2.triangulatePoints(P1, P2, pose_2d_r_one, pose_2d_l_one)
        # in milimeters
        pose_3d_one = pose_4d_one[0:3,:]/pose_4d_one[3,:]
        
        gt_kps.append(pose_3d_one)
    
    # Triangulation result
    gt_kps_3d = np.array(gt_kps)
    gt_kps_3d_matlab = gt_kps_3d.transpose(0,2,1)/1000
    
    # construct a dict that have all info: 2d pose left and right, and naive trigulation 3D pose, later also add the refine 3d pose
        
    pose_2d_l_all = np.array(pose_2d_l_all)
    pose_2d_r_all = np.array(pose_2d_r_all)
    

    pose_3d_refine = {}
    pose_3d_refine['frame_list_inter'] = frame_list_inter
    pose_3d_refine['pose_2d_l'] = pose_2d_l_all
    pose_3d_refine['pose_2d_r'] = pose_2d_r_all
    pose_3d_refine['pose_3d_tri'] = gt_kps_3d
    # 4xN array of reconstructed points in homogeneous coordinates, the 4th index is ones matrix, xyz in mm unit
    pose_3d_refine['pose_3d_tri_4'] = np.insert(gt_kps_3d, 3, values=1, axis=1)
    
    # 3d kps for matlab format
    pose_3d_refine['pose_3d_tri_matlab'] = gt_kps_3d_matlab
    
 

    project_2d_pose_r = project2d(P1, pose_3d_refine['pose_3d_tri'])
    project_2d_diff_r = project_2d_pose_r - pose_3d_refine['pose_2d_r']
    
    project_2d_pose_l = project2d(P2, pose_3d_refine['pose_3d_tri'])
    project_2d_diff_l = project_2d_pose_l - pose_3d_refine['pose_2d_l']
    
    pose_3d_refine['proj_2d_l'] = project_2d_pose_l
    pose_3d_refine['proj_2d_r'] = project_2d_pose_r
    
    
    pose_3d_refine['proj_2d_l_diff'] = project_2d_diff_l
    pose_3d_refine['proj_2d_r_diff'] = project_2d_diff_r
    
    pose_3d_refine['est_pose_3d_r'] = pose_3d_r['kps_3d']
    pose_3d_refine['frame_list_l'] = pose_3d_l['frame_list']
    pose_3d_refine['frame_list_r'] = pose_3d_r['frame_list']
    pose_3d_refine['frame_total'] = len(pose_2d_r)
    return pose_3d_refine




# %%
"""
MAIN STARTS HERE
"""
def get_videolabels(subject):  
    # torch.set_default_tensor_type(torch.FloatTensor)
    
    # joint_names = ["nose","left_eye","right_eye","left_ear","right_ear",
    #               "left_shoulder","right_shoulder","left_elbow","right_elbow",
    #               "left_wrist","right_wrist","left_hip","right_hip","left_knee",
    #               "right_knee","left_ankle","right_ankle"]
    
    kp_connections = [[0, 5], [0, 6], [5, 6],
                      [5, 7], [6, 8], [7, 9], [8, 10],
                      [5, 11], [6, 12],
                      [11,13], [12, 14], [13, 15], [14, 16]]
    global bone_head_idx, bone_tail_idx
    
    bone_head_idx = list(np.array(kp_connections).T[0])
    bone_tail_idx = list(np.array(kp_connections).T[1])
    
    # load files
    pose_2d_l, pose_2d_r, pose_3d_l, pose_3d_r, video_labels = loadfiles(subject)
    camera_params = getPmat()
    P1 = camera_params['cam1_proj_mat']
    P2 = camera_params['cam2_proj_mat']
    # get triangulation 3d points
    pose_3d_refine = get_gt_kps_3d(pose_2d_l, pose_2d_r, pose_3d_l, pose_3d_r, P1, P2)
    
    # plot to check the projected back 2D points
    # sanity_plot(pose_3d_refine)
    
    return pose_3d_refine, video_labels

# # %%
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument('-subj', '--subject', type=str, default = 'subject10',
#            help='Get 3D points for which subject')
#     main(**vars(p.parse_args()))
    


