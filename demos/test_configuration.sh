#!/bin/bash

respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
batch_size=1

# Optimization parameters
learning_rates=(0.1)
contact_treshholds=(0.2)
weights=(0.1 0.25 0.5 1.0)
optim_steps=100
optim_unit_grad=1
optim_anneal_lr=1

# Contact parameters
init_noise_scale=0.1

# Visualization parameters
visualize_sdf=0
load_cache=0

interaction_cfg_list=(
  './data/optim_interaction/cab_e/path2/run_8.json'
)

model_list=(
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

for interaction_cfg in "${interaction_cfg_list[@]}"; do
  interaction_name=$(basename "$interaction_cfg" .json)
  echo "[INFO] Running interaction: $interaction_name"

  for model in "${model_list[@]}"; do

    for optim_lr in "${learning_rates[@]}"; do
      echo "[INFO] Running optimization with learning rate: $optim_lr"

      for contact_thresh in "${contact_treshholds[@]}"; do
        echo "[INFO] Running with contact threshold: $contact_thresh"

        for weight_collision in "${weights[@]}"; do
          for weight_contact in "${weights[@]}"; do
            for weight_jerk in "${weights[@]}"; do
              for weight_skate in "${weights[@]}"; do

                echo "[INFO] Running with collision=$weight_collision, contact=$weight_contact, jerk=$weight_jerk, skate=$weight_skate"

                time python -m mld.optim_scene_mld \
                  --denoiser_checkpoint "$model" \
                  --interaction_cfg "$interaction_cfg" \
                  --optim_lr $optim_lr \
                  --optim_steps $optim_steps \
                  --batch_size $batch_size \
                  --guidance_param $guidance \
                  --respacing "$respacing" \
                  --export_smpl $export_smpl \
                  --use_predicted_joints $use_predicted_joints \
                  --optim_unit_grad $optim_unit_grad \
                  --optim_anneal_lr $optim_anneal_lr \
                  --weight_jerk $weight_jerk \
                  --weight_collision $weight_collision \
                  --weight_contact $weight_contact \
                  --weight_skate $weight_skate \
                  --contact_thresh $contact_thresh \
                  --load_cache $load_cache \
                  --init_noise_scale $init_noise_scale \
                  --visualize_sdf $visualize_sdf

                if [ $visualize_sdf -eq 1 ]; then
                  ffmpeg -framerate 10 -i "bin/imgs/sdf-points-0-%d.png" -c:v libx264 -pix_fmt yuv420p "bin/result/optim_${interaction_name}.mp4" -y
                  echo "[INFO] Video saved as 'bin/result/optim_${interaction_name}.mp4'."
                fi
              done
            done
          done
        done
      done
    done
  done
done
