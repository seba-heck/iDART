# iDART: Synthesizing Interactive Human Behavior
**Course group project for Digital Humans, ETH Zürich, FS2025.**

## Setup

This section describes the setup and installation for the code of the group project. See the description of DART ([Getting Started](./DART-README.md#getting-started)) for the complete setup.

### Requirements and Environment

This setup uses a conda environment. We recommend to use miniconda ([Miniconda - ANACONDA](https://www.anaconda.com/docs/getting-started/miniconda/main)).

```
conda env create -f environment.yml
conda activate iDART
```

The experimental setup used 2 Intel Xeon CPUs and 1 NVIDIA GTX 1080 Ti from the student cluster.

### Download Data and Model Checkpoints

The project depends on model checkpoints and data sets from DART and data for the body models. Please follow the links, download the material, unpack and merge it with this repository. 
- [DART data - Google Drive](https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps?usp=drive_link): folders can be copied into the root directory.
- [SMPL-X body model](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip): insert into data folder (exact structure below)
- [SMPL-H body model](https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz): insert into smplx folder (exact structure below)

<details>
  <summary> Root folder structure </summary>

  ```
  .
  ├── config_files
  ├── control
  ├── data             *new
  ├── data_loaders
  ├── data_scripts
  ├── demos
  ├── diffusion
  ├── environment.yml
  ├── evaluation
  ├── misc
  ├── mld
  ├── mld_denoiser    *new
  ├── model
  ├── mvae            *new
  ├── policy_train    *new
  ├── README.md
  ├── scenes
  ├── utils
  └── visualize
  ...
  ``` 
</details>

<details>
  <summary> Data folder structure </summary>

  ```
  data
  ├── action_statistics.json
  ├── fps_dict_all.json
  ├── fps_dict.json
  ├── hml3d_smplh
  │   └── seq_data_zero_male
  ├── inbetween
  │   └── pace_in_circles
  ├── joint_skin_dist.json
  ├── optim_interaction
  │   ├── climb_down.json
  │   └── sit.json
  ├── scenes
  │   └── demo
  ├── seq_data_zero_male
  │   ├── mean_std_h1_f1.pkl
  │   ├── mean_std_h2_f16.pkl
  │   ├── mean_std_h2_f8.pkl
  │   ├── train_text_embedding_dict.pkl
  │   └── val_text_embedding_dict.pkl
  ├── smplx_lockedhead_20230207                        *from other source
  │   └── models_lockedhead                            *unpack and move models here
  ├── stand_20fps.pkl
  ├── stand.pkl
  ├── test_locomotion
  │   ├── demo_walk.json
  │   ├── random.json
  │   ├── test_hop_long.json
  │   ├── test_run_long.json
  │   └── test_walk_long.json
  └── traj_test
      ├── dense_frame180_walk_circle
      ├── dense_frame180_wave_right_hand_circle
      ├── sparse_frame180_walk_square
      └── sparse_punch
  ```
</details>
  
<details>
  <summary> SMPL-X folder structure </summary>

  ```
  data/smplx_lockedhead_20230207/
  └── models_lockedhead
      ├── smplh                     *from MANO/SMPL-H
      │   ├── female
      │   ├── info.txt
      │   ├── LICENSE.txt
      │   ├── male
      │   └── neutral
      └── smplx                     *from SMPL-X
          ├── md5sums.txt
          ├── SMPLX_FEMALE.npz
          ├── SMPLX_MALE.npz
          └── SMPLX_NEUTRAL.npz
  ```
</details>

#### ⚠️ IMPORTANT
Use the correct names for the folders, especially for the SMPl-X folder, and be careful with data folder, it already contains some necessary files.

#### **Data Set Description**

**ds002748 (v.1.0.1):** Two sessions of resting state with closed eyes for patients with depression in treatment course (NFB, CBT or No treatment groups)

This dataset was used for the resting state network analysis for patients with mild depression. There are two session of RS with 2-month interval, 3 groups of patients: no treatment, CBT, or fmri-NFB treatment. A session consists of 100 dynamic scans with TR = 2.5 s and 25 slices.

https://openneuro.org/datasets/ds003007/versions/1.0.1

**ds002748 (v.1.0.5):** Resting state with closed eyes for patients with depression and healthy participants

This dataset was used for the resting state network analysis. There are 51 subjects with mild depression and 21 healthy controls. A session consist of 100 dynamic scans with TR = 2.5 s and 25 slices.

https://openneuro.org/datasets/ds002748/versions/1.0.5

### ⚠️ IMPORTANT
Only download the subjects `sub-51` up to `sub-72` to receive the healthy subjects.

### ⚠️ IMPORTANT
Only download the subjects `sub-51` up to `sub-72` to receive the healthy subjects.

### ⚠️ IMPORTANT
Only download the subjects `sub-51` up to `sub-72` to receive the healthy subjects.

## Guideline

Open Matlab and go to the project repository. Next, run following command. This will run the pipeline in right order.

```
for preprocessing:
run('src/fmri_prepro_depression.m')
run('src/fmri_prepro_healthy.m')

for statistical analysis:
run('src/rDCM_healthy_vs_depressed_analysis.m')
run('src/rDCM_three_group_treatment_analysis.m')
run('src/rDCM_treatment_pre_post_analysis.m')

for svm modeling:
run('src/svm_routine.m')

the rest of .m files are helper functions
```
### 📝 NOTE  
- Preprocessing fMRI data can take a **very long time**. On the local machine used, it took approximately **8–12 hours** to preprocess everything
- appendix folder contains 2 codes that were not used for the final conclusions because they didn't show significant results (hypotheses_overall_results.m, figures in that folder obtained from here) or very slow accuracy predictions (appendix/svm_treat_pred.m). Nevertheless, they are part of the analysis development and/or future research.

## Authors 

Areli Balderas Maza: abalderas@student.ethz.ch 
David Blickenstorfer: davidbl@student.ethz.ch 
Zhining Zhang: zhinzhang@student.ethz.ch

## Acknowledgment

The authors thanks Dr. Heinzle Jakob for his supervising during the project, Computational Psychiatry Lab for providing the rDCM toolbox, and the developers of SPM12 for providing neuroimaging tools.

## License
The code was developed at University of Zurich and ETH Zurich and is part of the course: **Translational Neuromodeling, 227-0973-00L**. 