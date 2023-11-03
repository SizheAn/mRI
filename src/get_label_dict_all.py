
import numpy as np
from read_cpl import get_videolabels
import os
import json
import pickle
import torch
from camera_calibrate import getPmat

subject_list = ['subject' + str(i) for i in range(1,21)]

for subject in subject_list:
    out_folder = '../label_dict/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # camera
    dict, video_labels = get_videolabels(subject)
    time_array = np.load('../rawdata/unixtime/' + subject + '_unix.npy', allow_pickle=True)
    num_frames_all = len(time_array)

    # init all keypoints file with ALL FRAMES, can have 0 at the beginning and end 
    num_dims = 3
    num_joints = 17
    rgb_kps_all = np.zeros([num_frames_all, num_dims, num_joints],
                    dtype=dict['est_pose_3d_r'].dtype)
    imu_kps_all = np.zeros([num_frames_all, num_dims, num_joints],
                    dtype=dict['est_pose_3d_r'].dtype)
    radar_kps_all = np.zeros([num_frames_all, num_dims, num_joints],
                    dtype=dict['est_pose_3d_r'].dtype)
    naive_kps_all = np.zeros([num_frames_all, num_dims, num_joints],
                    dtype=dict['est_pose_3d_r'].dtype)
    refined_kps_all = np.zeros([num_frames_all, num_dims, num_joints],
                dtype=dict['est_pose_3d_r'].dtype)         
    twod_l_kps_all = np.zeros([num_frames_all, num_dims-1, num_joints],
            dtype=dict['est_pose_3d_r'].dtype)
    twod_r_kps_all = np.zeros([num_frames_all, num_dims-1, num_joints],
            dtype=dict['est_pose_3d_r'].dtype)  
    


    
    # RGB model data
    rgb_frame_start = dict['frame_list_r'][0]
    rgb_frame_end = dict['frame_list_r'][-1]
    rgb_kps = dict['est_pose_3d_r'].transpose(0,2,1)
    rgb_kps_all[rgb_frame_start:rgb_frame_end+1] = rgb_kps
    # load radar data
    radar_data = np.load('../features/radar/' + subject + '_featuremap.npy').astype(np.float32).transpose(0,3,1,2)
    radar_data = torch.from_numpy(radar_data)

    # load refined_points
    refined_folder = '../aligned_data/pose_labels/'
    refined_label_name = refined_folder + subject + '.cpl'
    #read label
    with open(refined_label_name, 'rb') as fp:
        refined_label = pickle.load(fp)
    gt_frame_start = dict['frame_list_inter'][0]
    gt_frame_end = dict['frame_list_inter'][-1]
    refined_kps = refined_label['pose_3d_refined']/1000
    naive_kps = dict['pose_3d_tri']/1000
    refined_kps[:,0,:] = -refined_kps[:,0,:]
    refined_kps[:,1,:] = -refined_kps[:,1,:]
    naive_kps[:,0,:] = -naive_kps[:,0,:]
    naive_kps[:,1,:] = -naive_kps[:,1,:]
    refined_kps_all[gt_frame_start:gt_frame_end+1] = refined_kps
    naive_kps_all[gt_frame_start:gt_frame_end+1] = naive_kps



    # init radar CNN model
    torch.cuda.set_device(1)
    device = torch.device('cuda:1')
    radar_model = torch.load('model/mmWave/mmWave_protocol1.pkl')

    radar_model.eval()

    radar_kps = np.array(radar_model(radar_data.cuda()).detach().cpu()).reshape(-1,3,17)
    radar_kps[:,0,:] = -radar_kps[:,0,:]
    radar_kps[:,1,:] = -radar_kps[:,1,:]
    radar_frame_start = video_labels['T pose'][0]
    radar_frame_end = video_labels['walk'][1]
    radar_kps_all[radar_frame_start:radar_frame_end+1] = radar_kps
    # load imu data
    imu_data = torch.load('../features/imu/' + subject + '/acc_ori.pt')


    # init CNN model
    imu_model = torch.load('model/imu/imu_protocol1.pkl')

    imu_model.eval()
    imu_kps = np.array(imu_model(imu_data[:,None,:,:].cuda()).detach().cpu()).reshape(-1,3,17)
    imu_kps[:,0,:] = -imu_kps[:,0,:]
    imu_kps[:,1,:] = -imu_kps[:,1,:]
    imu_frame_start = video_labels['T pose'][0]
    imu_frame_end = video_labels['walk'][1]
    imu_kps_all[imu_frame_start:imu_frame_end+1] = imu_kps
    

    # 2d available frames
    twod_l_kps_all[dict['frame_list_l'][0]:dict['frame_list_l'][-1]+1] = dict['pose_2d_l']
    twod_r_kps_all[dict['frame_list_l'][0]:dict['frame_list_l'][-1]+1] = dict['pose_2d_r']

 
    # storing a new dict
    new_dict = {}
    # naive gt, refined gt, est rgb, radar, imu; rgb, radar, imu frame index
    
    new_dict['naive_gt_kps'] = naive_kps_all
    new_dict['refined_gt_kps'] = refined_kps_all
    new_dict['rgb_est_kps'] = rgb_kps_all
    new_dict['radar_est_kps'] = radar_kps_all
    new_dict['imu_est_kps'] = imu_kps_all
    new_dict['rgb_avail_frames'] = [rgb_frame_start, rgb_frame_end]
    new_dict['radar_avail_frames'] = [radar_frame_start, radar_frame_end]
    new_dict['imu_avail_frames'] = [imu_frame_start, imu_frame_end]
    new_dict['gt_avail_frames'] = [gt_frame_start, gt_frame_end]
    new_dict['2d_l_avail_frames'] = [dict['frame_list_l'][0], dict['frame_list_l'][-1]]
    new_dict['2d_r_avail_frames'] = [dict['frame_list_l'][0], dict['frame_list_l'][-1]]
    new_dict['pose_2d_l'] = twod_l_kps_all
    new_dict['pose_2d_r'] = twod_r_kps_all
    new_dict['video_label'] = video_labels
    new_dict['camera_matrix'] = getPmat()



    # np.save(subject + '_gt_kps_optimized.npy', pts_optim_matlab/1000)
    out_label_name = out_folder + subject + '_all_labels.cpl'
    #save label
    with open(out_label_name, 'wb') as fp:
        pickle.dump(new_dict, fp)



    print('Save labels for: ', subject)   