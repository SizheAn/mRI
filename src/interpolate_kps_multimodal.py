import argparse
import os
import torch
import json
import pickle
import numpy as np

from multimodal_CNN import CNN

def find_matchidx(arr1, arr2, idx):
    value = arr1[idx]
    diff = np.abs(arr2 - value)
    return np.argmin(diff)

def main(modality):
    # Check if CUDA is available and choose device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    subject_list = [f'subject{i}' for i in range(1,21)]

    for subject in subject_list:
        out_folder = f'../video_seg/{subject}/'

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # labels
        pose_file_name = '../aligned_data/pose_labels/' + subject + '.cpl'
        video_labels = pickle.load(open(f'../label_dict/{subject}_all_labels.cpl', 'rb'))['video_label']
        with open(pose_file_name, 'rb') as f:  
            dict = pickle.load(f)
    
        time_array = np.load(f'../rawdata/unixtime/{subject}_unix.npy', allow_pickle=True)

        if modality == 'imu':
            # load IMU data
            meta_data = torch.load(f'../features/imu/{subject}/acc_ori.pt')
        elif modality == 'radar':
            meta_data = np.load(f'../features/radar/{subject}_featuremap.npy').astype(np.float32).transpose(0,3,1,2)
            meta_data = torch.from_numpy(meta_data)
        elif modality == 'rgb':
            kps = dict['est_pose_3d_r']

        # initialize the model
        if modality == 'radar':
            model = CNN(channels=5, neurons=51, fc_input_size=6272).to(device)
        elif modality == 'imu':
            model = CNN(channels=1, neurons=51, fc_input_size=2304).to(device)
        
        if modality in ['radar', 'imu']:
            model.load_state_dict(torch.load(f"model/{modality}/{modality}_protocol2_datasplit1.pt"))
            model.eval()

            if modality == 'radar':
                kps = np.array(model(meta_data.cuda()).detach().cpu()).reshape(-1,3,17)
            elif modality == 'imu':
                kps = np.array(model(meta_data[:,None,:,:].cuda()).detach().cpu()).reshape(-1,3,17)


        if modality in ['radar', 'imu']:
            kps_frame_start = video_labels['T pose'][0]
            kps_frame_end = video_labels['walk'][1]
        else:
            kps_frame_start = dict['frame_list_r'][0]
            kps_frame_end = dict['frame_list_r'][-1]

        # get xp for np.interp
        time_array = (time_array - time_array[0])
        time_array_unit = time_array/time_array[-1]

        # get x for np.interp
        relative_time_array = time_array - time_array[0] # in miliseconds
        time_length_ms = np.max(relative_time_array)
        upsampled_frequency = 50
        interval_ms = 1000/upsampled_frequency
        upsampled_time_array = np.linspace(0,time_length_ms,round(time_length_ms/interval_ms)).astype('int')
        upsampled_time_array_unit = upsampled_time_array/upsampled_time_array[-1]

        # input dim / output dim
        num_frames_kps = kps.shape[0]
        num_joints = kps.shape[2]
        num_dims = kps.shape[1]

        upsampled_num_frames = len(upsampled_time_array_unit)

        interp_kps = np.zeros([upsampled_num_frames, num_dims, num_joints],
                            dtype=kps.dtype)

        # coordinates
        xp = time_array_unit
        x = upsampled_time_array_unit

        upsampled_start = find_matchidx(time_array_unit, upsampled_time_array_unit, kps_frame_start)
        upsampled_end = find_matchidx(time_array_unit, upsampled_time_array_unit, kps_frame_end)

        # 1D interpolation
        for idx in range(num_joints):
            for dim in range(num_dims):
                ifp = np.interp(x[upsampled_start:upsampled_end], xp[kps_frame_start:kps_frame_end+1], kps[:, dim, idx])
                # put the interpolated values back
                interp_kps[upsampled_start:upsampled_end, dim, idx] = ifp

        # new labels in mm seconds
        for key in video_labels.keys():
            video_labels[key][0] = int(upsampled_time_array[find_matchidx(time_array, upsampled_time_array, np.where(time_array == time_array[video_labels[key][0]])[0][0])])
            video_labels[key][1] = int(upsampled_time_array[find_matchidx(time_array, upsampled_time_array, np.where(time_array == time_array[video_labels[key][1]])[0][0])])
        
        # # save pose
        # if modality in ['radar', 'imu']:
        #     np.save(out_folder + f'kps3d_{modality}.npy', interp_kps.transpose(0,2,1))
        # # rgb
        # else:
        #     out_label_name = out_folder + 'labels.json'
        #     # save label
        #     with open(out_label_name,'w') as f:
        #         json.dump(video_labels,f) 
        #     # save pose
        #     np.save(out_folder + 'kps3d.npy', interp_kps)

        print(f"output shape is: {interp_kps.shape}")
        print(f"successfully upsampled for : {subject}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upsample modality data')
    parser.add_argument('--modality', type=str, default='imu', help='Modality to be upsampled,  ["imu","radar","rgb"]')
    args = parser.parse_args()
    main(args.modality)
