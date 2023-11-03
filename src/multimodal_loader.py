# -*- coding: utf-8 -*-
"""
Created on Mon May 10 00:06:28 2021

@author: sizhean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
import sys 

torch.set_default_tensor_type(torch.FloatTensor)


torch.manual_seed(1)  # reproducible
 
transform = transforms.Compose([
    transforms.ToTensor(), 
])

path = os.getcwd()
os.chdir(path)


class MyDataset():
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transforms = transform
    def __getitem__(self, index):
        datapoint= self.data[index, :, :] if self.data.ndim == 3 else self.data[index, :, :, :]
        datapoint = np.squeeze(datapoint)  
        labelpoint = self.label[index, :]  
        datapoint= self.transforms(datapoint)  
        return datapoint, labelpoint 
    def __len__(self):
        return self.data.shape[0]

def get_data_from_subject(subject_list, protocol, modality):
    # two protocols
    movement_list1 = ['pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5', 'pose_6', 'pose_7', 'pose_8', 'pose_9', 'pose_10', 'free_form', 'walk']
    movement_list2 = ['pose_1', 'pose_2', 'pose_3', 'pose_4', 'pose_5', 'pose_6', 'pose_7', 'pose_8', 'pose_9', 'pose_10']
    
    metadata_all = []
    label_all = []
    for subject in subject_list:
        pose_file_name = '../aligned_data/pose_labels/' + subject + '.cpl'
        video_labels = pickle.load(open(f'../label_dict/{subject}_all_labels.cpl', 'rb'))['video_label']
        with open(pose_file_name, 'rb') as f:  
            dict = pickle.load(f)
        meta_data = np.load('../features/' + modality + '/' + subject + '_featuremap.npy').astype(np.float32) if modality == 'radar' else torch.load('../features/imu/' + subject + '/acc_ori.pt')
        
        metadata_camera_offset = video_labels['T pose'][0]
        camera_pose_offset = dict['frame_list_inter'][0]

        if protocol == 2:
            metadata_subject = [meta_data[(video_labels[movement][0] - metadata_camera_offset):(video_labels[movement][1] - metadata_camera_offset + 1),:,:] for movement in movement_list2]
            label_subject = [dict['pose_3d_refined'][(video_labels[movement][0] - camera_pose_offset):(video_labels[movement][1] - camera_pose_offset + 1),:,:] for movement in movement_list2]
        else:
            metadata_subject = [meta_data[(video_labels[movement][0] - metadata_camera_offset):(video_labels[movement][1] - metadata_camera_offset + 1),:,:] for movement in movement_list1]
            label_subject = [dict['pose_3d_refined'][(video_labels[movement][0] - camera_pose_offset):(video_labels[movement][1] - camera_pose_offset + 1),:,:] for movement in movement_list1]

        metadata_subject = np.vstack(metadata_subject)
        label_subject = np.vstack(label_subject)

        label_subject = label_subject.reshape((len(label_subject), 17*3))/1000

        metadata_all.append(metadata_subject)
        label_all.append(label_subject)

        assert(len(metadata_subject) == len(label_subject))

    return np.vstack(metadata_all), np.vstack(label_all)


def get_data(subject_list, protocol, modality, random_seed):
    print(f'load from {modality} features')
    metadata, label = get_data_from_subject(subject_list, protocol, modality)
    dataset = MyDataset(metadata, label)
    
    batch_size = 128
    test_split = .2
    shuffle_dataset = True

    # Creating data indices for training and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = data.SubsetRandomSampler(train_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)

    train_loader =  data.DataLoader(dataset, batch_size=batch_size, sampler = train_sampler)
    test_loader =  data.DataLoader(dataset, batch_size=batch_size, sampler = test_sampler)    

    return train_loader, test_loader

def get_data_subject(test_subject_list, protocol, modality):
    print(f'load from {modality} features')
    subject_list = ['subject' + str(i) for i in range(1,21)]
    train_subject_list = list(set(subject_list).difference(set(test_subject_list)))

    train_data, train_label = get_data_from_subject(train_subject_list, protocol, modality)
    test_data, test_label = get_data_from_subject(test_subject_list, protocol, modality)

    train_dataset = MyDataset(train_data, train_label)
    test_dataset = MyDataset(test_data, test_label)
    
    batch_size = 128
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and test splits:
    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    train_indices = list(range(train_dataset_size))
    test_indices = list(range(test_dataset_size))
 
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    # Creating PT data samplers and loaders:
    train_sampler = data.SubsetRandomSampler(train_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)

    train_loader =  data.DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler)
    test_loader =  data.DataLoader(test_dataset, batch_size=batch_size, sampler = test_sampler)    

    return train_loader, test_loader
