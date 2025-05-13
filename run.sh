#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output=nvidia-smi.out
source ./demos/scene.sh
python -m visualize.vis_seq --add_floor 0 --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/sit_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.1_jerk0.1/sample_*.pkl'

