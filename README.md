# iDART - Interactive DART
## Synthesizing Interactive Human Behavior
Team 6 of course group project for Digital Humans, ETH Zürich FS2025.

## Notes
- <details>
    <summary> Download file structure </summary>
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
