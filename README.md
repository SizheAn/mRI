# mRI:
**Data repo for mRI: Multi-modal 3D Human Pose Estimation Dataset using mmWave, RGB-D, and Inertial Sensors**

**Demo in Project page: https://sizhean.github.io/mri**

**[Dataset download link in google drive](https://drive.google.com/file/d/1kR2U_omRkVTNkoetr7Akkorx5HfAvZ_C/view?usp=sharing)**

**Teaser: Try our action localization in action_localization folder!**

_Please note that we need to process the part of the data (**camera related modalities**) due to privacy-preserving protocol, which might delay the data release. The dataset (including **camera realted modalities**) will be fully open-sourced soon._

After unzip the dataset_release.zip, the folder structure should be like this: 

```
${ROOT}
|-- raw_data
|   |-- imu
|   |-- eaf_file
|   |-- radar
|   |-- unixtime
|   |-- videolabels
|-- aligned_data
|   |-- imu
|   |-- radar
|   |-- pose_labels
|-- features
|   |-- imu
|   |-- radar
|-- model
|   |-- imu
|   |   |-- results
|   |   |-- *.pkl
|   |-- mmWave
|   |   |-- results
|   |   |-- *.pkl
```

**_raw_data_** folder contains all raw_data before synchronization. It includes imu raw data, radar raw data, eaf annotations, unix timestamp from camera, and videolabels generated from the eaf file.

**_aligned_data_** folder contains all data after temporal alignment. It includes imu data, radar data, and the pose_labels. 
pose_labels for each subject contain following information:

'2d_l_avail_frames': available frames for 2d human detection, left camera  
'2d_r_avail_frames': available frames for 2d human detection, right camera  
'camera_matrix': camera parameters  
'gt_avail_frames': available frames for 3d human joints ground truth  
'imu_avail_frames': available frames for imu-estimated keypoints  
'imu_est_kps': imu-estimated keypoints  
'naive_gt_kps': naive triangulation keypoints  
'pose_2d_l': human 2d keypoints from left camera  
'pose_2d_r': human 2d keypoints from right camera  
'radar_avail_frames': available frames for radar-estimated keypoints  
'radar_est_kps': radar-estimated keypoints  
'refined_gt_kps': refined triangulation keypoints ground truth  
'rgb_avail_frames': available frames for rgb-estimated keypoints  
'rgb_est_kps': rgb-estimated keypoints  
'video_label': video action labels  

**_feature_** folder contains imu, radar features for deep learning models. The features are generated from the synced data.

Dimension of the radar feature is (frames, 14, 14, 5). The final 5 means x, y, z-axis coordinates, Doppler velocity, and intensity.

Dimension of the radar feature is (frames, 6, 12). 6 is the number of IMUs and 12 is flattened 3x3 rotation and 3 accelerations.

**_model_** folder contains the pretrained model .pkl files and and results.
