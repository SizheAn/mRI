# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:42:19 2022

@author: sizhe-admin
"""

# read the output pose .cpl file and refine triangulated 3D points
#%%
import pickle
import argparse
import numpy as np

from read_cpl import get_videolabels
from utils import get_mpjpe_rgb


# %%
"""
MAIN STARTS HERE
"""
def main():  
        
    # COCO format
    # joint_names = ["nose","left_eye","right_eye","left_ear","right_ear",
    #               "left_shoulder","right_shoulder","left_elbow","right_elbow",
    #               "left_wrist","right_wrist","left_hip","right_hip","left_knee",
    #               "right_knee","left_ankle","right_ankle"]

    # human3.6M estimate format
    # joint_names = ["center_hip","right_hip","right_knee","right_ankle","left_hip","left_knee","left_ankle",
    #               "mid_section","neck","nose","head","left_shoulder","left_elbow","left_hand", 
    #               "right_shoulder","right_elbow","right_hand"]

    # 13 common joint: left_shoulder, left_elbow, left_wrist/hand, left_hip, left_knee, left_ankle, 
    #               right_shoulder, right_elbow, right wrist/hand, right_hip, right_knee, right_ankle,
    #               nose
    #               

    # hard-coded the common_joint idx for COCO and human3.6M
    joint_coco_idx = [5,7,9,11,13,15,6,8,10,12,14,16,0]
    joint_human_idx = [11,12,13,4,5,6,14,15,16,1,2,3,9]

    # subject_list = ['subject' + str(i) for i in range(1,21)]

    # #for paper 
    all_test_subject_list = [['subject17', 'subject13', 'subject11', 'subject15'],
    ['subject9', 'subject7', 'subject20', 'subject8'],
    ['subject3', 'subject16', 'subject7', 'subject2']]

    total_error = []
    for subject_list in all_test_subject_list:
        error_group = []
        for subject in subject_list:

            pose_file_name = '../aligned_data/pose_labels/' + subject + '.cpl'
            # get triangulation 3d points
            _, video_labels = get_videolabels(subject)
            with open(pose_file_name, 'rb') as f:  
                pose_3d_refine = pickle.load(f)

            start_idx_gt = video_labels['T pose'][0]
            end_idx_gt = video_labels['walk'][1]
            all_kps_pre = np.zeros((pose_3d_refine['frame_total'], 3,17))
            all_kps_gt = np.zeros((pose_3d_refine['frame_total'], 3,17))

            all_kps_pre[pose_3d_refine['frame_list_r'], :, :] = pose_3d_refine['est_pose_3d_r'].transpose(0,2,1)
            all_kps_gt[pose_3d_refine['frame_list_inter'], :, :] = pose_3d_refine['pose_3d_refined']

            pre_3d_kpt = 1000*all_kps_pre[start_idx_gt:end_idx_gt, :, :]
            gt_3d_kpt = all_kps_gt[start_idx_gt:end_idx_gt, :, :]
            
            pre_3d_kpt[:,0,:] = -pre_3d_kpt[:,0,:]
            pre_3d_kpt[:,1,:] = -pre_3d_kpt[:,1,:]
            # plot_3dpose(gt_3d_kpt, pre_3d_kpt, 500)

            # # # for paper evaluate, 80%
            # np.random.seed(4312)
            # np.random.shuffle(pre_3d_kpt)
            # np.random.seed(4312)
            # np.random.shuffle(gt_3d_kpt)
            # pre_3d_kpt = pre_3d_kpt[0:round(0.2*len(pre_3d_kpt))]
            # gt_3d_kpt = gt_3d_kpt[0:round(0.2*len(gt_3d_kpt))]

            p1_error, p2_error = get_mpjpe_rgb(pre_3d_kpt, gt_3d_kpt)
            error_group.append([p1_error, p2_error])
        total_error.append(np.mean(np.array(error_group),0))
    total_error.append(np.mean(total_error, 0))
    total_error.append(np.std(total_error, 0))
    # plot_3dpose(gt_3d_kpt, pre_3d_kpt, 500)
    print('finish')

# %%
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(**vars(p.parse_args()))