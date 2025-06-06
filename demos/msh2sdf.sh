respacing='ddim10'
guidance=5
export_smpl=1
use_predicted_joints=1
optim_lr=0.01
#optim_lr=0.1
optim_steps=100
optim_unit_grad=1
optim_anneal_lr=1

weight_jerk=0.1
weight_collision=1.0
weight_contact=0.1
weight_skate=0.0
contact_thresh=0.00
init_noise_scale=0.1

visualize_sdf=0

load_cache=0
interaction_cfg_list=(
# './data/optim_interaction/foodlab/walk_test.json'
# './data/optim_interaction/foodlab/sit_8k.json'
# './data/optim_interaction/cab_benches/walk_test.json'
# './data/optim_interaction/cab_e/walk.json'
'./data/optim_interaction/dlab_0215/sit-chair.json'
# './data/optim_interaction/seminar_h53/walk_test.json'
)

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

columns=$(tput cols)

for interaction_cfg in "${interaction_cfg_list[@]}"; do
  interaction_name=$(basename "$interaction_cfg" .json)
  printf "%-${columns}s\n" | tr ' ' "="
  printf " > ${interaction_name}\n"

  for model in "${model_list[@]}"; do
    time python -m mld.optim_scene_msh2sdf --denoiser_checkpoint "$model" --interaction_cfg "$interaction_cfg" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size 2 --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk --weight_collision $weight_collision  --weight_contact $weight_contact  --weight_skate $weight_skate  --contact_thresh $contact_thresh  --load_cache $load_cache  --init_noise_scale $init_noise_scale
    time python -m mld.optim_scene_mld --denoiser_checkpoint "$model" --interaction_cfg "$interaction_cfg" --optim_lr $optim_lr --optim_steps $optim_steps --batch_size 1 --guidance_param $guidance --respacing "$respacing" --export_smpl $export_smpl  --use_predicted_joints $use_predicted_joints  --optim_unit_grad $optim_unit_grad  --optim_anneal_lr $optim_anneal_lr  --weight_jerk $weight_jerk --weight_collision $weight_collision  --weight_contact $weight_contact  --weight_skate $weight_skate  --contact_thresh $contact_thresh  --load_cache $load_cache  --init_noise_scale $init_noise_scale --visualize_sdf $visualize_sdf
  done
done