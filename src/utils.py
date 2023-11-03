# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:42:19 2022

@author: sizhe-admin
"""

# metrics and plotting utils for mri project
#%%
import numpy as np
import matplotlib.pyplot as plt

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    # return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
    return np.linalg.norm(predicted - target, axis=2)

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)

def plot_3dpose(gt_3d_kpt, pre_3d_kpt, frame_num):
    
    
    xs1 = -gt_3d_kpt[frame_num,0,:]
    ys1 = -gt_3d_kpt[frame_num,1,:]
    zs1 = gt_3d_kpt[frame_num,2,:]

    xs2 = -pre_3d_kpt[frame_num,0,:]
    ys2 = -pre_3d_kpt[frame_num,1,:]
    zs2 = pre_3d_kpt[frame_num,2,:]



    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1,projection='3d')

    for i in range(len(xs2)):
        ax.scatter(xs1[i], ys1[i], zs1[i], marker='o')
        ax.text(xs1[i], ys1[i], zs1[i],  '%s' % (str(i)), size=20, zorder=1, 
        color='k') 

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.azim = -90

    # ax.elev = -90
    ax.azim = -90

    ax.elev = 100
    plt.title("tri")

    ax = fig.add_subplot(1,2,2,projection='3d')


    # for i in range(len(m)): #plot each point + it's index as text above
    #     ax.scatter(m[i,0],m[i,1],m[i,2],color='b') 
    #     ax.text(m[i,0],m[i,1],m[i,2],  '%s' % (str(i)), size=20, zorder=1,  
    #     color='k') 

    for i in range(len(xs2)):
        ax.scatter(xs2[i], ys2[i], zs2[i], marker='o')
        ax.text(xs2[i], ys2[i], zs2[i],  '%s' % (str(i)), size=20, zorder=1, 
        color='k') 

    # ax.scatter(xs2, ys2, zs2, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.azim = -90

    ax.elev = 100
    plt.title("est")
    plt.tight_layout()
    plt.savefig("./test.jpg")

def plot_3dpose_one(gt_3d_kpt, pre_3d_kpt):
    
    
    xs1 = -gt_3d_kpt[0,:]
    ys1 = -gt_3d_kpt[1,:]
    zs1 = gt_3d_kpt[2,:]

    xs2 = -pre_3d_kpt[0,:]
    ys2 = -pre_3d_kpt[1,:]
    zs2 = pre_3d_kpt[2,:]



    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1,projection='3d')

    for i in range(len(xs1)):
        ax.scatter(xs1[i], ys1[i], zs1[i], marker='o')
        ax.text(xs1[i], ys1[i], zs1[i],  '%s' % (str(i)), size=20, zorder=1, 
        color='k') 
    # ax.scatter(xs1, ys1, zs1, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.azim = -90

    # ax.elev = -90
    ax.azim = -90

    ax.elev = 100
    plt.title("tri")

    ax = fig.add_subplot(1,2,2,projection='3d')

    for i in range(len(xs2)):
        ax.scatter(xs2[i], ys2[i], zs2[i], marker='o')
        ax.text(xs2[i], ys2[i], zs2[i],  '%s' % (str(i)), size=20, zorder=1, 
        color='k') 
    # ax.scatter(xs2, ys2, zs2, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.azim = -90

    ax.elev = 100
    plt.title("est")
    plt.tight_layout()
    plt.savefig("./test.jpg")
#MPJPE

def get_mpjpe_rgb(pre_3d_kpt, gt_3d_kpt):
    # hard-coded the common_joint idx for COCO and human3.6M
    joint_coco_idx = [5,7,9,11,13,15,6,8,10,12,14,16,0]
    # joint_human_idx = [14,15,16,1,2,3,11,12,13,4,5,6,9]
    # somehow human index is flipped
    joint_human_idx = [11,12,13,4,5,6,14,15,16,1,2,3,9]

    num_evaluate = len(joint_coco_idx)
    sample_num = len(gt_3d_kpt)

    pre_3d_kpt = pre_3d_kpt[:,:, joint_human_idx].transpose(0,2,1)
    gt_3d_kpt = gt_3d_kpt[:,:, joint_coco_idx].transpose(0,2,1)


    pre_3d_kpt_root = (pre_3d_kpt[:,3,:] + pre_3d_kpt[:,9,:])/2
    # gt root-relative error, subtract avg of left and right hip
    gt_3d_kpt_root = (gt_3d_kpt[:,3,:] + gt_3d_kpt[:,9,:])/2

    # agign to root
    pre_3d_kpt = pre_3d_kpt - pre_3d_kpt_root[:, np.newaxis,:]
    gt_3d_kpt = gt_3d_kpt - gt_3d_kpt_root[:, np.newaxis,:]

    # plot_3dpose(gt_3d_kpt.transpose(0,2,1), pre_3d_kpt.transpose(0,2,1), 500)

    p1_error = np.mean(mpjpe(pre_3d_kpt, gt_3d_kpt), 1)
    p2_error = np.mean(p_mpjpe(pre_3d_kpt, gt_3d_kpt), 1)

    threshold = 1000
    p1_good = np.where(p1_error < threshold)
    p2_good = np.where(p2_error < threshold)

    p1_error = p1_error[p1_good[0]].copy()
    p2_error = p2_error[p2_good[0]].copy()

    return np.mean(p1_error), np.mean(p2_error)

def get_mpjpe(pre_3d_kpt, gt_3d_kpt):


    pre_3d_kpt = pre_3d_kpt.transpose(0,2,1)
    gt_3d_kpt = gt_3d_kpt.transpose(0,2,1)


    pre_3d_kpt_root = (pre_3d_kpt[:,3,:] + pre_3d_kpt[:,9,:])/2
    # gt root-relative error, subtract avg of left and right hip
    gt_3d_kpt_root = (gt_3d_kpt[:,3,:] + gt_3d_kpt[:,9,:])/2

    # agign to root
    pre_3d_kpt = pre_3d_kpt - pre_3d_kpt_root[:, np.newaxis,:]
    gt_3d_kpt = gt_3d_kpt - gt_3d_kpt_root[:, np.newaxis,:]

    p1_error = np.mean(mpjpe(pre_3d_kpt, gt_3d_kpt), 1)
    p2_error = np.mean(p_mpjpe(pre_3d_kpt, gt_3d_kpt), 1)

    threshold = 1000
    p1_good = np.where(p1_error < threshold)
    p2_good = np.where(p2_error < threshold)

    p1_error = p1_error[p1_good[0]].copy()
    p2_error = p2_error[p2_good[0]].copy()

    return np.mean(p1_error), np.mean(p2_error)
