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
from read_cpl import get_videolabels
import matplotlib.pyplot as plt
import math


def get_joint_angle(kpt, joint_idx):
    # we want left_shoulder, left_elbow, left_wrist/hand, left_hip, left_knee, left_ankle, 
    #               right_shoulder, right_elbow, right wrist/hand, right_hip, right_knee, right_ankle,
    coco_idx = [5,7,9,11,13,15,6,8,10,12,14,16]
    human_idx = [11,12,13,4,5,6,14,15,16,1,2,3]

    kpt = kpt[:,:,joint_idx]
    num_frames = len(kpt)
    all_joints = np.zeros((num_frames, 4))
    # left upper, left low, right upper, right low
    for i in range(num_frames):
        joint_group0 = kpt[i, :, 0:3]
        joint_group1 = kpt[i, :, 3:6]
        joint_group2 = kpt[i, :, 6:9]
        joint_group3 = kpt[i, :, 9:12]

        joint_0 = cal_angle(joint_group0[:,0], joint_group0[:,1], joint_group0[:,2])
        joint_1 = cal_angle(joint_group1[:,0], joint_group1[:,1], joint_group1[:,2])
        joint_2 = cal_angle(joint_group2[:,0], joint_group2[:,1], joint_group2[:,2])
        joint_3 = cal_angle(joint_group3[:,0], joint_group3[:,1], joint_group3[:,2])

        all_joints[i][0] = joint_0
        all_joints[i][1] = joint_1
        all_joints[i][2] = joint_2
        all_joints[i][3] = joint_3

    return all_joints

def cal_angle(point_a, point_b, point_c):
    """

    :param point_a、point_b、point_c: 数据类型为list,二维坐标形式[x、y]或三维坐标形式[x、y、z]
    :return: 返回角点b的夹角值


    数学原理：
    设m,n是两个不为0的向量，它们的夹角为<m,n> (或用α ,β, θ ,..,字母表示)

    1、由向量公式：cos<m,n>=m.n/|m||n|

    2、若向量用坐标表示，m=(x1,y1,z1), n=(x2,y2,z2),

    则,m.n=(x1x2+y1y2+z1z2).

    |m|=√(x1^2+y1^2+z1^2), |n|=√(x2^2+y2^2+z2^2).

    将这些代入②得到：

    cos<m,n>=(x1x2+y1y2+z1z2)/[√(x1^2+y1^2+z1^2)*√(x2^2+y2^2+z2^2)]

    上述公式是以空间三维坐标给出的，令坐标中的z=0,则得平面向量的计算公式。

    两个向量夹角的取值范围是：[0,π].

    夹角为锐角时，cosθ>0；夹角为钝角时,cosθ<0.

    """
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        # print("坐标点为3维坐标形式")
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0,0,0  # 坐标点为2维坐标形式，z 坐标默认值设为0
        # print("坐标点为2维坐标形式，z 坐标默认值设为0")

    # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
    x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)

    # 两个向量的夹角，即角点b的夹角余弦值
    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) *(math.sqrt(x2**2 + y2**2 + z2**2))) # 角点b的夹角余弦值
    if cos_b > 1:
        cos_b = 1
    elif cos_b < -1:
        cos_b = -1
    B = math.degrees(math.acos(cos_b)) # 角点b的夹角值
    return B

def get_joint_velocity(kpt, joint_idx):
    kpt = kpt[:,:,joint_idx]
    num_frames = len(kpt)
    kpt_0 = kpt[0:num_frames-2]
    kpt_1 = kpt[1:num_frames-1]
    kpt_diff = np.abs(kpt_0 - kpt_1)
    kpt_velocity = kpt_diff/0.1
    return kpt_velocity

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

    subject_list = ['subject' + str(i) for i in range(1,21)]

    # #for paper 
    # all_test_subject_list = [['subject17', 'subject13', 'subject11', 'subject15'],
    # ['subject9', 'subject7', 'subject20', 'subject8'],
    # ['subject3', 'subject16', 'subject7', 'subject2']]
    # all_test_subject_list = [['subject' + str(i) for i in range(1,21)],
    # ['subject' + str(i) for i in range(1,21)],
    # ['subject' + str(i) for i in range(1,21)]]
    total_error_angle = []
    total_error_velocity = []

    seed_list = [665,666,667]
    # for subject_list in all_test_subject_list:
    for random_seed in seed_list:
        error_angle_group = []
        error_velocity_group = []
        for subject in subject_list:

            label_file_name = '../label_dict/' + subject + '_all_labels.cpl'
            with open(label_file_name, 'rb') as f:  
                all_label = pickle.load(f)

            print('load label from: ', subject)

            # eval_frames = [all_label['video_label']['pose_1'][0], all_label['video_label']['pose_10'][1]]
            # refined_gt_kps = all_label['refined_gt_kps'][eval_frames[0]:eval_frames[1]]
            # rgb_est_kps = all_label['rgb_est_kps'][eval_frames[0]:eval_frames[1]]
            # radar_est_kps = all_label['radar_est_kps'][eval_frames[0]:eval_frames[1]]
            # imu_est_kps = all_label['imu_est_kps'][eval_frames[0]:eval_frames[1]]

            eval_frames = [all_label['video_label']['pose_1'][0], all_label['video_label']['pose_10'][1]]
            
            refined_gt_kps = all_label['refined_gt_kps'][eval_frames[0]:eval_frames[1]]
            rgb_est_kps = all_label['rgb_est_kps'][eval_frames[0]:eval_frames[1]]
            radar_est_kps = all_label['radar_est_kps'][eval_frames[0]:eval_frames[1]]
            imu_est_kps = all_label['imu_est_kps'][eval_frames[0]:eval_frames[1]]

            np.random.seed(random_seed)
            refined_gt_kps = refined_gt_kps[0:round(0.2*len(refined_gt_kps))]
            np.random.seed(random_seed)
            rgb_est_kps = rgb_est_kps[0:round(0.2*len(rgb_est_kps))]
            np.random.seed(random_seed)
            radar_est_kps = radar_est_kps[0:round(0.2*len(radar_est_kps))]
            np.random.seed(random_seed)
            imu_est_kps = imu_est_kps[0:round(0.2*len(imu_est_kps))]

            gt_joint_angle = get_joint_angle(refined_gt_kps, joint_coco_idx)
            rgb_joint_angle = get_joint_angle(rgb_est_kps, joint_human_idx)
            radar_joint_angle = get_joint_angle(radar_est_kps, joint_coco_idx)
            imu_joint_angle = get_joint_angle(imu_est_kps, joint_coco_idx)

            rgb_joint_error = np.mean(np.abs(gt_joint_angle - rgb_joint_angle), axis = 0)
            radar_joint_error = np.mean(np.abs(gt_joint_angle - radar_joint_angle), axis = 0)
            imu_joint_error = np.mean(np.abs(gt_joint_angle - imu_joint_angle), axis = 0)

            gt_joint_velocity = get_joint_velocity(refined_gt_kps, joint_coco_idx)
            rgb_joint_velocity = get_joint_velocity(rgb_est_kps, joint_human_idx)
            radar_joint_velocity = get_joint_velocity(radar_est_kps, joint_coco_idx)
            imu_joint_velocity = get_joint_velocity(imu_est_kps, joint_coco_idx)            

            rgb_velocity_error = np.mean(np.mean(np.abs(gt_joint_velocity - rgb_joint_velocity), axis = 0), axis = 0)
            radar_velocity_error = np.mean(np.mean(np.abs(gt_joint_velocity - radar_joint_velocity), axis = 0), axis = 0)
            imu_velocity_error = np.mean(np.mean(np.abs(gt_joint_velocity - imu_joint_velocity), axis = 0), axis = 0)


            # # # for paper evaluate, 80%
            # np.random.seed(4312)
            # np.random.shuffle(pre_3d_kpt)
            # np.random.seed(4312)
            # np.random.shuffle(gt_3d_kpt)
            # pre_3d_kpt = pre_3d_kpt[0:round(0.2*len(pre_3d_kpt))]
            # gt_3d_kpt = gt_3d_kpt[0:round(0.2*len(gt_3d_kpt))]

            np.random.seed(random_seed)
            error_angle_group.append(np.array([radar_joint_error, rgb_joint_error, imu_joint_error]))
            error_velocity_group.append(np.array([radar_velocity_error, rgb_velocity_error, imu_velocity_error]))

        total_error_angle.append(np.mean(np.array(error_angle_group),0))
        total_error_velocity.append(np.mean(np.array(error_velocity_group),0))

    mean_angle = np.mean(np.array(total_error_angle), 0)
    std_angle = np.std(np.array(total_error_angle), 0)

    mean_velocity = np.mean(np.array(total_error_velocity), 0)*1000
    std_velocity = np.std(np.array(total_error_velocity), 0)*1000
    # plot_3dpose(gt_3d_kpt, pre_3d_kpt, 500)
    print('finish')

# %%
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(**vars(p.parse_args()))