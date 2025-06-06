#!/bin/bash
#
#SBATCH --job-name=DH6-testConfig
#
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=24:00:00
#SBATCH --output=cabE_path2_run8_weight_at_lr_0p1_ct_0p2.out

# conda activate DART

# RUN OPTIM
# source ./demos/leglos.sh
source ./demos/test_configuration.sh

# VISUALIZE
# python -m visualize.vis_seq --add_floor 0 --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/sit_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.1_jerk0.1/sample_*.pkl'

