# -*- coding: utf-8 -*-
"""
Created on Fri May  6 00:24:40 2022

@author: sizhe-admin
"""
# this code get camera parameters from matlab and convert it to opencv version..
import cv2
import numpy as np

def getPmat():
    #%%
    # # P = K[R|t] P: projection matrix, K: intrinsic matrix, R: rotation matrix, T: translation matrix
    # # https://stackoverflow.com/questions/39447624/back-projecting-3d-world-point-to-new-view-image-plane

    # # R|T
    # extrinsic = np.column_stack((R1, np.array([0,0,0]).T))
    # # K[R|t]
    # points = np.matmul(left_camera_matrix, extrinsic)


    # camera 1 on the right side, but it is the coordinate origin
    # camera 1 intrinsic
    # [1.030792174583942e+03,0.175946363713597,9.839463572673553e+02;0,1.034034428253797e+03,5.533467762302013e+02;0,0,1]
    camera1_intrinsic = np.array([[1.030792174583942e+03,0.175946363713597,9.839463572673553e+02],
                                    [0,1.034034428253797e+03,5.533467762302013e+02],
                                    [0,0,1]])  
    
    R_1 = np.eye(3)
    T_1 = np.array([0,0,0]).T
    extrinsic_1 = np.column_stack((R_1, T_1))
    
    P1_new = np.matmul(camera1_intrinsic, extrinsic_1)
    
    # camera 2 intrinsic
    # [1.025798745272310e+03,0.683062692332954,9.512388877206313e+02;0,1.030167601128334e+03,5.348743322203921e+02;0,0,1]
    camera2_intrinsic = np.array([[1.025798745272310e+03,0.683062692332954,9.512388877206313e+02],
                                [0,1.030167601128334e+03,5.348743322203921e+02],
                                [0,0,1]])  
    # camera 2 rotation
    # [0.999570578533327,0.004722676856741,-0.028919800377179;-0.004938025703504,0.999960578444242,-0.007379529768819;0.028883809179203,0.007519167557925,0.999554494605739]
    # camera 2 translation
    # [2.500309846364905e+02,1.598544695919455,-2.243813907811485]
    R_2 = np.array([[0.999570578533327,0.004722676856741,-0.028919800377179],
                    [-0.004938025703504,0.999960578444242,-0.007379529768819],
                    [0.028883809179203,0.007519167557925,0.999554494605739]]).T
    
    T_2 = np.array([2.500309846364905e+02,1.598544695919455,-2.243813907811485]).T

    
    extrinsic_2 = np.column_stack((R_2, np.matmul(R_2,T_2)))
    
    # It is the same as what we calculate from matlab
    P2_new = np.matmul(camera2_intrinsic, extrinsic_2)
    
    # # p2_new from matlab function P = cameraMatrix(cameraParams,rotationMatrix,translationVector)
    # P2_new = np.array([[997.851832411082, -11.4020804950308, 980.449216942469, 254348.159179637],
    #                 [-10.6033102262774, 1026.17986926161, 542.382045625767, 446.610489124271],
    #                 [-0.0289198003771792, -0.00737952976881942, 0.999554494605739, -2.24381390781148]])
    
    camera_params = {}
    camera_params['cam1_intrinsic'] = camera1_intrinsic
    camera_params['cam1_extrinsic'] = extrinsic_1
    camera_params['cam1_R'] = R_1
    camera_params['cam1_T'] = T_1
    camera_params['cam1_proj_mat'] = P1_new
    
    camera_params['cam2_intrinsic'] = camera2_intrinsic
    camera_params['cam2_extrinsic'] = extrinsic_2
    camera_params['cam2_R'] = R_2
    camera_params['cam2_T'] = T_2
    camera_params['cam2_proj_mat'] = P2_new

    
    return camera_params
# print(Q)


