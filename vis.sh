#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=48:00:00
#SBATCH --output=vis.out

python -m visualize.vis_seq --add_floor 1 --body_type smplx --seq_path './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/inbetween/repeatseed/scale0.1_floor0.0_jerk0.0_use_pred_joints_ddim10_pace_in_circles*15_guidance5.0_seed0/*.pkl'

