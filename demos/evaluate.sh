respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
batch_size=8
optim_lr=0.01
#optim_lr=0.1
optim_steps=100
optim_unit_grad=1
optim_anneal_lr=1

weight_jerk=0.1
weight_collision=0.1
weight_contact=0.1
weight_skate=0.0
contact_thresh=0.00
init_noise_scale=0.1

visualize_sdf=0

load_cache=0

interaction_cfg_list=(
  './data/optim_interaction/foodlab/walk_test.json'
  './data/optim_interaction/foodlab/walk_test.json'
  './data/optim_interaction/foodlab/sit_8k.json'
  './data/optim_interaction/foodlab/sit_8k.json'
  './data/optim_interaction/cab_benches/walk_test.json'
  './data/optim_interaction/cab_benches/walk_test.json'
  './data/optim_interaction/cab_e/walk.json'
  './data/optim_interaction/cab_e/walk.json'
  './data/optim_interaction/dlab_0215/sit-chair.json'
  './data/optim_interaction/dlab_0215/sit-chair.json'
  './data/optim_interaction/seminar_h53/walk_test.json'
  './data/optim_interaction/seminar_h53/walk_test.json'
  # './data/optim_interaction/seminar_h53/walk_test.json'
  # './data/optim_interaction/seminar_h53/walk_test.json'
  # './data/optim_interaction/seminar_h53/walk_test.json'
  # './data/optim_interaction/seminar_h53/walk_test.json'
  # './data/optim_interaction/seminar_h53/walk_test.json'
  # './data/optim_interaction/seminar_h53/walk_test.json'
)

pkl_file_list=(
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/foodlab_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/foodlab_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/foodlab_sit-8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/foodlab_sit-8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/benches_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/benches_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/cab_e_walk_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/cab_e_walk_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/cnb_dlab_0215_sit_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/cnb_dlab_0215_sit_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'
  # './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.0_jerk0.1/volSMPL_sample_0.pkl'
  # './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.1_jerk0.1/volSMPL_sample_0.pkl'
  # './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision0.5_jerk0.1/volSMPL_sample_0.pkl'
  # './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/volSMPL_sample_0.pkl'
  # './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision2.0_jerk0.1/volSMPL_sample_0.pkl'
  # './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision5.0_jerk0.1/volSMPL_sample_0.pkl'
)

# Ensure both lists have the same length
if [ ${#interaction_cfg_list[@]} -ne ${#pkl_file_list[@]} ]; then
  echo "Error: The number of interaction configuration files and .pkl files must match."
  exit 1
fi

# Iterate through both lists simultaneously
for i in "${!interaction_cfg_list[@]}"; do
  interaction_cfg="${interaction_cfg_list[$i]}"
  pkl_file="${pkl_file_list[$i]}"

  interaction_name=$(basename "$interaction_cfg" .json)

  time python -m evaluation.scene_interaction --interaction_cfg "$interaction_cfg" --smpl_file "$pkl_file" --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk --weight_collision $weight_collision  --weight_contact $weight_contact  --weight_skate $weight_skate  --contact_thresh $contact_thresh  --load_cache $load_cache  --init_noise_scale $init_noise_scale --visualize_sdf $visualize_sdf

  if [ $visualize_sdf -eq 1 ]; then
    printf "%-${columns}s\n" | tr ' ' "-"
    ffmpeg -framerate 10 -i "bin/imgs/sdf-points-eval-%d.png" -c:v libx264 -pix_fmt yuv420p "bin/result/optim_${interaction_name}.mp4" -y -hide_banner
    echo "[INFO] Video saved as 'bin/result/optim_${interaction_name}.mp4'."
    rm -r bin/imgs/sdf-points-eval-*.png
  fi

done