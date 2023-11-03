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
import torch
import torch.optim as optim
from torch.autograd import Variable
from scipy.signal import savgol_filter
from read_cpl import get_videolabels
import time


#%%choose one frame to plot projected 2d points
def sanity_plot(pose_3d_refine):
    # Todo: set proper frame_num variable to make it for every frame 
    
    file1_name = '../temp_workspace/left/000300.jpg'
    file2_name = '../temp_workspace/right/000300.jpg'
    frame1 = cv2.imread(file1_name)
    frame2 = cv2.imread(file2_name)
    
    num_frame = file1_name[-7:-4]
    
    frame_offset = pose_3d_refine['frame_list_inter'][0]
    #GT
    plt.imshow(frame1[:,:,[2,1,0]])
    plt.scatter(pose_3d_refine['pose_2d_l'][int(num_frame) - frame_offset][0,:], pose_3d_refine['pose_2d_l'][int(num_frame) - frame_offset][1,:], s = 3)
    plt.scatter(pose_3d_refine['proj_2d_l'][int(num_frame) - frame_offset][0,:], pose_3d_refine['proj_2d_l'][int(num_frame) - frame_offset][1,:], s = 3, c ='red')
    plt.show()
    
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(pose_3d_refine['pose_2d_r'][int(num_frame) - frame_offset][0,:], pose_3d_refine['pose_2d_r'][int(num_frame) - frame_offset][1,:], s = 3)
    plt.scatter(pose_3d_refine['proj_2d_r'][int(num_frame) - frame_offset][0,:], pose_3d_refine['proj_2d_r'][int(num_frame) - frame_offset][1,:], s = 3, c ='red')
    plt.show()


#%% get bone length

def loss_bonebias(pts, bone_median):

    bone_head = pts[:,:,bone_head_idx]
    bone_tail = pts[:,:,bone_tail_idx]
    bone_length = torch.linalg.norm(bone_tail-bone_head, axis=1)
    bone_length_bias = torch.sub(bone_length,bone_median)
    
    return bone_length_bias

# %%
# loss function L = ||P1*3D_left - 2D_left|| + ||P2*3D_right - 2D_right|| + || bone_bias || + || P(t+1) - P(t) ||
def skeleton_loss(pts, P1, P2, ql, qr, bone_median):

    # step1: 2, 3, ... n
    # step2: 1, 2, ... n-1
    # diff, 2-1, 3-2, ... n - (n-1)
    pts_step1 = pts[1:,:,:]
    pts_step2 = pts[:-1,:,:]
    pts_diff = torch.linalg.norm(pts_step1  - pts_step2, dim = 1)

    # cat the tensor to be homogenous
    pts_homo = torch.cat((pts, torch.ones((pts.shape[0], 1, pts.shape[2]))), dim = 1)
    project_2d_pose_l = torch.matmul(P2, pts_homo)
    project_2d_pose_valid_l = torch.divide(project_2d_pose_l[:,0:2,:], project_2d_pose_l[:,2,:][:,np.newaxis])
    project_2d_diff_l = project_2d_pose_valid_l - ql
    
    project_2d_pose_r = torch.matmul(P1, pts_homo)
    # TODO: add eps
    project_2d_pose_valid_r = torch.divide(project_2d_pose_r[:,0:2,:], project_2d_pose_r[:,2,:][:,np.newaxis])
    project_2d_diff_r = project_2d_pose_valid_r - qr
    
    # MSE
    # compo1 = torch.pow(project_2d_diff_l, 2)
    # compo2 = torch.pow(project_2d_diff_r, 2)
    # compo3 = torch.pow(loss_bonebias(pts_homo, bone_median), 2)
    
    
    # MAE
    compo1 = torch.abs(project_2d_diff_l)
    compo2 = torch.abs(project_2d_diff_r)
    compo3 = torch.abs(loss_bonebias(pts_homo, bone_median))
    compo4 = pts_diff
    
    # loss_joint, loss1, loss2, loss3
    return torch.mean(compo1) + torch.mean(compo2) + torch.mean(compo3) + torch.mean(compo4), torch.mean(compo1), torch.mean(compo2), torch.mean(compo3), torch.mean(compo4)

# %%
"""
MAIN STARTS HERE
"""
def main(num_iteration, learning_rate):  
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # joint_names = ["nose","left_eye","right_eye","left_ear","right_ear",
    #               "left_shoulder","right_shoulder","left_elbow","right_elbow",
    #               "left_wrist","right_wrist","left_hip","right_hip","left_knee",
    #               "right_knee","left_ankle","right_ankle"]
    out_folder = '../aligned_data/pose_labels/'

    subject_list = ['subject' + str(i) for i in range(1,21)]

    # subject = 'subject1'
    
    # soft_constriants = [[0, 5], [0, 6]]
    kp_connections = [[5, 6],
                      [5, 7], [6, 8], [7, 9], [8, 10],
                      [5, 11], [6, 12],
                      [11,13], [12, 14], [13, 15], [14, 16]]
    global bone_head_idx, bone_tail_idx
    
    bone_head_idx = list(np.array(kp_connections).T[0])
    bone_tail_idx = list(np.array(kp_connections).T[1])
    

    camera_params = getPmat()
    P1 = camera_params['cam1_proj_mat']
    P2 = camera_params['cam2_proj_mat']

    for subject in subject_list:
        # get triangulation 3d points
        pose_3d_refine, _ = get_videolabels(subject)
        
        # # plot to check the projected back 2D points
        # sanity_plot(pose_3d_refine)
        
        # tensorize all
        # T_pts = Variable(torch.tensor(pose_3d_refine['pose_3d_tri_4']/1000).float(), requires_grad = True)
        T_pts = Variable(torch.tensor(pose_3d_refine['pose_3d_tri']).float(), requires_grad = True)
        T_P1 = Variable(torch.tensor(P1).float())
        T_P2 = Variable(torch.tensor(P2).float())
        T_ql = Variable(torch.tensor(pose_3d_refine['pose_2d_l']).float())
        T_qr = Variable(torch.tensor(pose_3d_refine['pose_2d_r']).float())

        
        bone_head = T_pts[:,:,bone_head_idx]
        bone_tail = T_pts[:,:,bone_tail_idx]
        bone_length = torch.linalg.norm(bone_tail-bone_head, axis=1)
        T_bone_length_median = torch.median(bone_length, 0)[0].detach()




        
        loss_list = []
        optimizer = optim.SGD([T_pts], lr=learning_rate) # We want to optimize T_pts

        start_time = time.time()
        #calculate for iterations
        for i in range(num_iteration):
            optimizer.zero_grad()
            loss_joint, loss1, loss2, loss3, loss4 = skeleton_loss(T_pts, T_P1, T_P2, T_ql, T_qr, T_bone_length_median)
            if i == 0:
                print("Initial loss is: ", loss_joint.item())
            loss_list.append([loss_joint.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()])
            #loss_joint for optimizing both, loss1 or loss2 if only optimizing one of them
            loss_joint.backward()

            optimizer.step()
            if i % (num_iteration-1) == 0:
                with torch.no_grad():
                    loss_joint,loss1,loss2,loss3,loss4 = skeleton_loss(T_pts, T_P1, T_P2, T_ql, T_qr, T_bone_length_median)
                print(subject + " loss is: ", loss_joint.item())
        
        end_time = time.time()
        print("total time is: ", end_time - start_time)
        loss_all = np.array(loss_list)
        
        # plt.plot(loss_all[:,0], color = 'r',label="loss_joint")
        # plt.plot(loss_all[:,1], color = 'y',label="loss1")
        # plt.plot(loss_all[:,2], color = 'c',label="loss2")
        # plt.plot(loss_all[:,3], color = 'g',label="loss3")
        # plt.xlabel("iterations")
        # plt.ylabel("Loss")
        # plt.legend(loc = "best")
        # plt.show()

        pts_optim = np.array(T_pts.detach())
        pts_optim_matlab = pts_optim.transpose(0,2,1)

        pose_3d_refine['pose_3d_refined'] = pts_optim
        pose_3d_refine['pose_3d_refined_matlab'] = pts_optim_matlab/1000
        
        # np.save(subject + '_gt_kps_optimized.npy', pts_optim_matlab/1000)
        out_label_name = out_folder + subject + '.cpl'
        #save label
        with open(out_label_name, 'wb') as fp:
            pickle.dump(pose_3d_refine, fp)
        
        print('Save labels for: ', subject)

# %%
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-iteration', '--num_iteration', type=int, default = 2000,
           help='Run the optimization for how many iterations')
    p.add_argument('-lr', '--learning_rate', type=float, default = 100000,
       help='The learning rate')
    main(**vars(p.parse_args()))