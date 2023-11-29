## MRI Dataset Simple Guide

### Part 1: Environment Requirements
This section is for demo. The whole requirement is listed in the `requirements.txt` file. You can install all the requirements by running the commands.


```python
import json
import numpy as np
import pickle
import cv2
import os
from camera_calibrate import getPmat
import matplotlib.pyplot as plt
import argparse
import numpy as np
from read_cpl import get_videolabels, project2d
from scipy import signal
import time

```

    

### Part 2: Data Structure

Here, we'll outline the structure of the MRI dataset, including both aligned data and raw rgb data. The dataset is organized as follows:

```
├── aligned feature data
│   ├── mmWave
│   ├── IMUs
│   ├── RGB
├── Raw RGB Data
│   ├── left camera
│   ├── right camera
```

#### Aligned Feature Data


```python
your_data_path = '/disk/san49/Dataset/'
```


```python
labels = pickle.load(open(your_data_path+'label_dict/subject1_all_labels.cpl', 'rb'))
radar_features = np.load(your_data_path+'dataset_release/features/radar/subject1_featuremap.npy')
pose_file_name = your_data_path+'aligned_data/pose_labels/' + 'subject1' + '.cpl'
with open(pose_file_name, 'rb') as f:  
    dict = pickle.load(f)
```

##### dataset details


```python
labels.keys(),dict.keys()
```




    (dict_keys(['naive_gt_kps', 'refined_gt_kps', 'rgb_est_kps', 'radar_est_kps', 'imu_est_kps', 'rgb_avail_frames', 'radar_avail_frames', 'imu_avail_frames', 'gt_avail_frames', '2d_l_avail_frames', '2d_r_avail_frames', 'pose_2d_l', 'pose_2d_r', 'video_label', 'camera_matrix']),
     dict_keys(['frame_list_inter', 'pose_2d_l', 'pose_2d_r', 'pose_3d_tri', 'pose_3d_tri_4', 'pose_3d_tri_matlab', 'proj_2d_l', 'proj_2d_r', 'proj_2d_l_diff', 'proj_2d_r_diff', 'est_pose_3d_r', 'frame_list_l', 'frame_list_r', 'frame_total', 'pose_3d_refined', 'pose_3d_refined_matlab']))




```python
labels['naive_gt_kps'].shape
```




    (6529, 3, 17)




```python
radar_features.shape
```




    (6384, 14, 14, 5)




```python
dict['est_pose_3d_r'].shape
```




    (6465, 17, 3)




```python
# IMU feature data, reference:https://arxiv.org/abs/2105.04605
interpolation = np.load(your_data_path+'video_seg/subject1/kps3d_imu.npy')
interpolation.shape
```




    (37998, 17, 3)



##### rgb data


```python
from camera_calibrate import getPmat
from read_cpl import get_videolabels, project2d
subject = 'subject1'
# please modify the path to your own data for function loadfiles in read_cpl.py
dict, video_labels = get_videolabels(subject)
pose_2d_l = dict['pose_2d_l'].transpose(0,2,1)
pose_2d_r = dict['pose_2d_r'].transpose(0,2,1)

points_2d = np.stack((pose_2d_r, pose_2d_l))
points_3d_init = dict['pose_3d_tri'].transpose(0,2,1)
```

#### Raw rgb data


```python
your_raw_data_path = "/disk/xxiong52/Dataset/rawdata/camera_blursubject1_0427_blurred/" # change to your own path
```


```python
for root, dirs, files in os.walk(your_raw_data_path):
    for name in files:
        if name.endswith(".mp4"):
            print(name)
```

    blurred_subject1_color0.mp4
    blurred_subject1_color1.mp4
    


```python
file_path = root+files[0]
video = cv2.VideoCapture(file_path)
print("video get:"+os.path.basename(file_path))
```

    video get:blurred_subject1_color0.mp4
    


```python
# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
codec = int(video.get(cv2.CAP_PROP_FOURCC))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width, height, fps, frame_count, codec
```




    (1920, 1080, 30.0, 6411, 877677894)




```python
# one frame in our dataset (T-pose)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_count//2)
ret, frame = video.read()
video.release()
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title(f"Middle Frame (Frame {frame_count//2})")
plt.axis('off')  # Turn off axis labels
plt.show()
```


    
![png](_data_template_files/_data_template_20_0.png)
    


## Part 3: function note
```
├── Data processing functions
│   ├── mmWave --> get_radarfeature.py  --> please load from multimodal_loader.py
│   ├── IMUs --> from aligned data[1] --> please load from multimodal_loader.py
│   ├── RGB --> from aligned data[2] --> please load from multimodal_loader.py
├── Label processing functions
│   ├── get_label_dict_all.py --> get all the labels aligned with lowest frequency
├── 3D pose estimation functions
│   ├── camera_calibrate.py --> get camera intrinsic parameters from matlab and convert it to opencv
│   ├── evaluate_3dpose.py --> read the output mmwave .cpl file and refine triangulated 3D points with mpjpe loss
│   ├── evaluate_jointangles.py --> read the output imu .cpl file and refine triangulated 3D points with mpjpe loss
│   ├── refine_3dpoints.py --> get 3D points from rgb with skeleton loss
│   ├── interpolate_kps_multimodal.py --> interpolate 3D points from modalities [mmwave, imu, rgb]
│   ├── triangulate_3dpoints.py --> triangulate 3D points from mmwave and imu data
│   ├── multimodal_train.py --> train or validate the model with argparse
├── Utils
│   ├── get_label_dict_all.py --> get all the labels and save to dict
│   ├── get_radarfeatures.py --> process mmwave data and save to dict
│   ├── multimodal_CNN.py --> models for encoding mmwave and imu data
│   ├── multimodal_loader.py --> build train and val dataloader from aligned data
│   ├── pytorchtools.py --> early stopping and save model
│   ├── utils.py --> some useful functions like mpjpe loss and plot


```

### Part 4: Demos
Please refer to test.py for refine 3d points from rgb data.


