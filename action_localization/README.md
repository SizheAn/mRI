# ActionFormer: Localizing Moments of Actions with Transformers
See README_ActionfFormer.md for a quick overview of the code.

## Installation
* Follow INSTALL.md for installing necessary dependencies and compiling the code.

## Experiments on mRI dataset
**Download Pose Data and Annotations**
* Download *mri_pose_data.zip* from [this Google Drive link](https://drive.google.com/file/d/1ve8y46bS17QxNQZL1tH-bgqSFmyaA1FH/view?usp=sharing).
* The file includes pose data in npz format, and action annotations in json format (similar to ActivityNet annotation format).

**Details**: The 3D pose are estimated from a RGB camera, IMU sensors, or a mmWave radar. The results are interpolated at **50Hz**.

**Unpack Pose Data and Annotations**
* Unpack the file under *./data* (or elsewhere and link to *./data*).
* The folder structure should look like
```
This folder
│   README.md
│   ...
│
└───data/
│    └───mri/
│    │	 └───annotations
│    │	 └───pose_features
│    └───...
|
└───libs
│
│   ...
```

**Training and Evaluation**
* To generate all experiments (3 modalities + 4 combinations across 3 splits and 2 protocols), run
```shell
python ./tools/gen_exps.py ./configs/ref_cfg.yaml ./configs/exp_cfg.yaml
```

* This will generate all config files (42 experiments in total) and a bash script under *./exp_configs*. To run all experiments, run
```shell
sh ./exp_configs/run_all_exps.sh
```
Results will be saved as txt files under *./ckpt*.

Here is an example for training with 3D poses from RGB frames on the first split using protocol 1.
* Train the model. This will create a experiment folder under *./ckpt* that stores training config, logs, and checkpoints.
```shell
python ./train.py ./exp_configs/rgb_s1_p1.yaml --output test -p 2
```
* [Optional] Monitor the training using TensorBoard
```shell
tensorboard --logdir=./ckpt/rgb_s1_p1_test/logs
```
* Evaluate the trained model.
```shell
python ./eval.py ./exp_configs/rgb_s1_p1.yaml ./ckpt/rgb_s1_p1_test/
```
* Training should take a few minutes at most with expected mAP around 93%.


## Contact
Yin Li (yin.li@wisc.edu)


## References
If you are using our code, please consider citing our paper.
```
@article{zhang2022actionformer,
  title={ActionFormer: Localizing Moments of Actions with Transformers},
  author={Zhang, Chenlin and Wu, Jianxin and Li, Yin},
  journal={arXiv preprint arXiv:2202.07925},
  year={2022}
}
```
