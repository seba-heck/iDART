from __future__ import annotations

import os
import pdb
import random
import time
from typing import Literal
from dataclasses import dataclass, asdict, make_dataclass
#sys.path.append(os.path.abspath(os.path.join(os.pa)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import pickle
import json
import copy
import trimesh
import time

from model.mld_denoiser import DenoiserMLP, DenoiserTransformer
from model.mld_vae import AutoMldVae
from data_loaders.humanml.data.dataset import WeightedPrimitiveSequenceDataset, SinglePrimitiveDataset
from utils.smpl_utils import *
from utils.misc_util import encode_text, compose_texts_with_and
from pytorch3d import transforms
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.resample import create_named_schedule_sampler

from mld.train_mvae import Args as MVAEArgs
from mld.train_mvae import DataArgs, TrainArgs
from mld.train_mld import DenoiserArgs, MLDArgs, create_gaussian_diffusion, DenoiserMLPArgs, DenoiserTransformerArgs
from mld.rollout_mld import load_mld, ClassifierFreeWrapper

from VolumetricSMPL import attach_volume

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

debug = 0

@dataclass
class OptimArgs:
    seed: int = 0
    torch_deterministic: bool = True
    device: str = "cuda"
    save_dir = './bin/output'

    denoiser_checkpoint: str = ''

    respacing: str = 'ddim10'
    guidance_param: float = 5.0
    export_smpl: int = 0
    zero_noise: int = 0
    use_predicted_joints: int = 0
    batch_size: int = 1

    optim_lr: float = 0.01
    optim_steps: int = 300
    optim_unit_grad: int = 1
    optim_anneal_lr: int = 1
    weight_jerk: float = 0.0
    weight_collision: float = 0.0
    weight_contact: float = 0.0
    weight_skate: float = 0.0
    floor_contact_loss: int = 0
    load_cache: int = 0
    contact_thresh: float = 0.03
    init_noise_scale: float = 1.0

    interaction_cfg: str = './data/optim_interaction/climb_up_stairs.json'

    visualize_sdf: int = 0


import torch.nn.functional as F
def calc_point_sdf(scene_assets, points):
    device = points.device
    scene_sdf_config = scene_assets['scene_sdf_config']
    scene_sdf_grid = scene_assets['scene_sdf_grid']
    sdf_size = scene_sdf_config['size']
    sdf_scale = scene_sdf_config['scale']
    sdf_scale = torch.tensor(sdf_scale, dtype=torch.float32, device=device).reshape(1, 1, 1)  # [1, 1, 1]
    sdf_center = scene_sdf_config['center']
    sdf_center = torch.tensor(sdf_center, dtype=torch.float32, device=device).reshape(1, 1, 3)  # [1, 1, 3]
    batch_size, num_points, _ = points.shape
    # convert to [-1, 1], here scale is (1.6/extent) proportional to the inverse of scene size, https://github.com/wang-ps/mesh2sdf/blob/1b54d1f5458d8622c444f78d4477f600a6fe50e1/example/test.py#L22
    points = (points - sdf_center) * sdf_scale  # [> B, num_points, 3]
    sdf_values = F.grid_sample(scene_sdf_grid.unsqueeze(0),  # [> B, 1, size, size, size]
                               points[:, :, [2, 1, 0]].view(batch_size, num_points, 1, 1, 3),
                               padding_mode='border',
                               align_corners=True
                               ).reshape(batch_size, num_points)
    # print('sdf_values', sdf_values.shape)
    sdf_values = sdf_values / sdf_scale.squeeze(-1)  # [> B, P], scale back to the original scene size
    return sdf_values

def sample_scene_points(scene_assets, B=8, T=98, s=4, rand_idxs=False):
    """
    Transforms the scene points into a tensor of shape [B, T, 3].

    Args:
        scene_assets: dict containing scene information, including "scene_points".
        B: Batch size.
        T: Number of time steps.

    Returns:
        A tensor of shape [B, T, 3].
    """
    all_points = scene_assets["scene_points"]  # Shape: [N, 3]
    # print("all_points shape", all_points.shape)

    # get scene points
    # bb_min = smpl_output.vertices.min(1).values.reshape(1, 3)
    # bb_max = smpl_output.vertices.max(1).values.reshape(1, 3)

    # inds = (scene_vertices >= bb_min).all(-1) & (scene_vertices <= bb_max).all(-1)
    # if not inds.any():
    #     return None
    # points = scene_vertices[inds]
    # points = points.float().reshape(1, -1, 3)  # add batch dimension
    # return points
    # Ensure the number of points matches B * T
    # Ensure the number of points matches B * T

    # repeat points for each time step
    stacked_pointspoints = all_points.unsqueeze(0).repeat(B, T, 1, 1)  # Shape: [T, N, 3]

    # Generate subset time frames indices
    if rand_idxs:
        indices = torch.randint(0, T, (T//s,))
    else:
        indices = torch.arange(0, T, step=s)
    # else:
    #     regular_indices = torch.arange(0, T, step=2*s)
    #     random_indices = torch.randint(0, T, (T//(2*s),))

    #     # Combine and remove duplicates
    #     indices = torch.cat([regular_indices, random_indices]).unique()
    
    t_ = len(indices)

    # select subset of time steps
    points = stacked_pointspoints[:,indices,:,:]  # Shape: [T//s, N, 3]
    # points = points.permute(2,0,1,3)
    points = points.reshape(B*t_, -1, 3)  # Shape: [B * T//s, N, 3]

    # print("Transformed points shape:", points.shape)
    return points, indices

def plot_sampled_points_with_sdf(points, sdf_values, smpl_output, B=8, T=98, title="Sampled Points with SDF Values"):
    """
    Plots the sampled points as a scatter plot, with SDF values as colors.

    Args:
        points (torch.Tensor): Tensor of shape [N, 3], containing the 3D coordinates of the points.
        sdf_values (torch.Tensor): Tensor of shape [N], containing the SDF values for the points.
        title (str): Title of the plot.
    """
    # Convert tensors to numpy arrays for plotting
    points_np = points.cpu().detach().numpy()
    sdf_values_np = sdf_values.cpu().detach().numpy()
    smpl_output_np = smpl_output.cpu().detach().numpy()

    for b_ in range(B):
        for t_ in range(T):
            # Create a 3D scatter plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot with SDF values as colors
            scatter = ax.scatter(points_np[t_,:, 0], points_np[t_,:, 1], points_np[t_,:, 2], c=sdf_values_np[-(1+b_),t_,:], cmap='viridis', s=5, alpha=0.4)

            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label("SDF Values")

            # scatter plot of SMPL body
            ax.scatter(smpl_output_np[-(1+b_),t_,:,0],smpl_output_np[-(1+b_),t_,:,1],smpl_output_np[-(1+b_),t_,:,2],c='black', s=10, alpha=1.0)

            # Set plot labels and title
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(title)

            # Ensure equally spaced axes
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            z_limits = ax.get_zlim()

            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]
            z_range = z_limits[1] - z_limits[0]
            max_range = max(x_range, y_range, z_range)

            # Center the axes
            x_middle = (x_limits[0] + x_limits[1]) / 2
            y_middle = (y_limits[0] + y_limits[1]) / 2
            z_middle = (z_limits[0] + z_limits[1]) / 2

            ax.set_xlim([x_middle - max_range / 2, x_middle + max_range / 2])
            ax.set_ylim([y_middle - max_range / 2, y_middle + max_range / 2])
            ax.set_zlim([z_middle - max_range / 2, z_middle + max_range / 2])
            
            # Set the angle of view
            ax.view_init(elev=40, azim=10)

            # Show the plot
            plt.tight_layout()
            plt.savefig(f"bin/imgs/sdf-points-{b_}-{t_}.png")
            plt.close()

    print(f'[Done] Plotting SDF-Points [bin/imgs/sdf-points-{B}-{T}.png]')

def calc_vol_sdf(scene_assets, motion_sequences, transf_rotmat, transf_transl, plot=False):
    """
    Calculate the signed distance function (SDF) values for the scene points.

    Args:
        scene_assets: dict containing scene information, including "scene_points".
        motion_sequences: dict containing motion sequences.
        transf_rotmat: rotation matrix for transformation.
        transf_transl: translation vector for transformation.
    Returns:
        A tensor of shape [B, t_, N] containing SDF values.
    Dimensions:
        B:  Batch size.
        T:  Number of time frames (for each batch).
        N:  Number of points in the scene.
        t_: Number of time frames selected from the scene points evaluation.
    """
    B, T, J, _ = motion_sequences['joints'].shape
    # print('B, T, J:', B, T, J)
    # print('Device:', transf_rotmat.device)

    transf_rotmat = transf_rotmat.to(device)
    transf_transl = transf_transl.to(device)
    # print("transf_rotmat", transf_rotmat.shape)
    # print("transf_transl", transf_transl.shape)

    # get body pose parameterscon
    betas = motion_sequences['betas'].reshape(B * T, 10).to(device=device) #body shape coeffs, don't rotate! 
    # > (B, T, 3, 3) -> (B, 3)
    global_orient = motion_sequences['global_orient'].to(device=device) #global rotation of root joint
    #  (B, T, 21, 3, 3)-> (B, (J*3))
    body_pose = motion_sequences['body_pose'].to(device=device) #relative rotations of other joints
    # (B, T, 3) -> (B, 3)
    transl = motion_sequences['transl'].to(device=device) #global translation of the root
    
    # print("betas shape", betas.shape)
    # print("global_orient shape", global_orient.shape)
    # print("body_pose shape", body_pose.shape)
    # print("transl shape", transl.shape)

    #Rotation Transformations
    # transf_rotmat.permute(0,2,1)
    #transform (multiply with transf_rotmat)
    # global_orient = torch.einsum('bij,btjk->btik', transf_rotmat, global_orient)
    # # body_pose = torch.einsum('bik,btjkl->btjil', transf_rotmat.permute(0,2,1), body_pose)
    # transl = torch.einsum('bij,btk->bti', transf_rotmat, transl)# + transf_transl

    #convert back to axis-angle
    global_orient = matrix_to_axis_angle(global_orient) 
    body_pose = matrix_to_axis_angle(body_pose)#.view(B, T * 21, 3, 3)).view(B, T, 21, 3) 
    # transl =  (transf_rotmat @ transl.transpose(-1, -2)).transpose(-1, -2)  + transf_transl# (B, T, 3)

    #reshape for required input size for SMPL-X
    global_orient = global_orient.reshape(B * T, 3).to(device=device)
    body_pose = body_pose.reshape(B * T, 21, 3).to(device=device)
    transl = transl.reshape(B * T, 3).to(device=device)

    assert global_orient.shape[0] == B * T and global_orient.shape[1] == 3  , f"global_orient shape or device mismatch: {global_orient.shape}/{(B * T, 3)}, {global_orient.device}/{device}"
    assert body_pose.shape[0] == B * T     and body_pose.shape[1] == 21     , f"body_pose shape or device mismatch: {body_pose.shape}/{(B * T, 21 * 3)}, {body_pose.device}"
    assert transl.shape[0] == B * T        and transl.shape[1] == 3         , f"transl shape or device mismatch: {transl.shape}/{(B * T, 3)}, {transl.device}"
    assert betas.shape[0] == B * T         and betas.shape[1] == 10         , f"betas shape or device mismatch: {betas.shape}/{(B * T, 10)}, {betas.device}"

    # get smplx model
    body_model = vol_body_model_dict[gender].to(device=device)

    # Debugging: Check devices of all tensors
    # print(f"Device check:\n betas={betas.device},\n global_orient={global_orient.device},\n body_pose={body_pose.device},\n transl={transl.device}")
    # print("global_orient final shape", global_orient.shape)
    # print("body_pose final shape", body_pose.shape)
    # print("nbr joints body model", body_model.NUM_BODY_JOINTS)

    # get "current" smpl body model
    smpl_output = body_model(betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             transl=transl,
                             return_verts=True, return_full_pose=True)
    # print("smpl_output shape", smpl_output.joints.shape)
    # print("smpl_output", type(smpl_output))

    # get scene points
    scene_points, idxs = sample_scene_points(scene_assets,1,T,s=6)
    # scene_points = scene_points.to(device=device)  # shape: [t, N, 3]

    # get nbr of (random) selected time frames
    t_ = len(idxs)

    # get SMPL body_model parameters
    smpl_output_full_pose = smpl_output.full_pose.reshape(B,T,-1,3)
    smpl_output_vertices = smpl_output.vertices.reshape(B,T,-1,3)
    smpl_output_joints = smpl_output.joints.reshape(B,T,-1,3)
    

    # body_model.volume.encode_body(subset_smpl_output)
    # print(body_model.volume.impl_code["bbox_min"].shape)
    # print(body_model.volume.impl_code["bbox_max"].shape)
    # print(body_model.volume.impl_code["bbox_size"].shape)
    # print(body_model.volume.impl_code["bbox_center"].shape)

    # query sampled points for sdf values
    # selfpen_loss, _collision_mask = body_model.volume.collision_loss(scene_points, smpl_output, ret_collision_mask=True)
    # sdf_values = body_model.volume.query_fast(scene_points, subset_smpl_output)

    # query sampled points for each batch
    sdf_values = torch.zeros(B, t_, scene_assets['nbr_points'], device=device)
    for i_ in range(B):
        # select specific smpl_output for each batch
        subset_smpl_output = smpl_output
        subset_smpl_output.full_pose = smpl_output_full_pose[0, idxs, ...].reshape(t_,-1,3)
        subset_smpl_output.vertices =  smpl_output_vertices[0, idxs, ...].reshape(t_,-1,3)
        subset_smpl_output.joints =    smpl_output_joints[0, idxs, ...].reshape(t_,-1,3)

        # Validate subset vertices
        if torch.isnan(subset_smpl_output.vertices).any() or torch.isinf(subset_smpl_output.vertices).any():
            print(f"[ERROR] Subset vertices for batch {i_} contain NaN or Inf values.")
            # subset_smpl_output.vertices = torch.nan_to_num(subset_smpl_output.vertices, nan=0.0, posinf=0.0, neginf=0.0)
            continue

        # query sampled points for sdf values
        sdf_values[i_,...] = body_model.volume.query_fast(scene_points, subset_smpl_output)

    # print(sdf_values.shape)

    if plot==True:
        plot_sampled_points_with_sdf(scene_points, sdf_values, smpl_output_vertices[:, idxs, ...], B=1, T=t_)

    return sdf_values # shape: [B, t_, N]

def calc_jerk(joints):
    vel = joints[:, 1:] - joints[:, :-1]  # --> > B x T-1 x 22 x 3
    acc = vel[:, 1:] - vel[:, :-1]  # --> > B x T-2 x 22 x 3
    jerk = acc[:, 1:] - acc[:, :-1]  # --> > B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> > B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1, 2])  # --> > B, Get the max of the jerk across all joints and frames

    return jerk.mean()

def get_text_promts(text_prompt):
    texts = []
    if ',' in text_prompt:  # contain a time line of multipel actions
        num_rollout = 0
        for segment in text_prompt.split(','):
            action, num_mp = segment.split('*')
            action = compose_texts_with_and(action.split(' and '))
            texts = texts + [action] * int(num_mp)
            num_rollout += int(num_mp)
    else:
        action, num_rollout = text_prompt.split('*')
        action = compose_texts_with_and(action.split(' and '))
        num_rollout = int(num_rollout)
        for _ in range(num_rollout):
            texts.append(action)
    return texts,num_rollout

def optimize(history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask):
    texts,num_rollout = get_text_promts(text_prompt)
    all_text_embedding = encode_text(dataset.clip_model, texts, force_empty_zero=True).to(dtype=torch.float32,
                                                                                      device=device)

    def rollout(noise, history_motion_tensor, transf_rotmat, transf_transl):
        motion_sequences = None
        history_motion = history_motion_tensor
        for segment_id in range(num_rollout):
            text_embedding = all_text_embedding[segment_id].expand(batch_size, -1)  # [> B, 512]
            guidance_param = torch.ones(batch_size, *denoiser_args.model_args.noise_shape).to(device=device) * optim_args.guidance_param
            y = {
                'text_embedding': text_embedding,
                'history_motion_normalized': history_motion,
                'scale': guidance_param,
            }

            x_start_pred = sample_fn(
                denoiser_model,
                (batch_size, *denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={'y': y},
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=False,
                noise=noise[segment_id],
            )  # [> B, T=1, D]
            # x_start_pred = x_start_pred.clamp(min=-3, max=3)
            # print('x_start_pred:', x_start_pred.mean(), x_start_pred.std(), x_start_pred.min(), x_start_pred.max())
            latent_pred = x_start_pred.permute(1, 0, 2)  # [T=1, > B, D]
            future_motion_pred = vae_model.decode(latent_pred, history_motion, nfuture=future_length,
                                                       scale_latent=denoiser_args.rescale_latent)  # [> B, F, D], normalized

            future_frames = dataset.denormalize(future_motion_pred)
            new_history_frames = future_frames[:, -history_length:, :]

            """transform primitive to world coordinate, prepare for serialization"""
            if segment_id == 0:  # add init history motion
                future_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
            future_feature_dict = primitive_utility.tensor_to_dict(future_frames)
            future_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :future_length, :] if segment_id > 0 else betas[:, :primitive_length, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'joints']:
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)  # [> B, T, ...]

            """update history motion seed, update global transform"""
            history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
            history_feature_dict.update(
                {
                    'transf_rotmat': transf_rotmat,
                    'transf_transl': transf_transl,
                    'gender': gender,
                    'betas': betas[:, :history_length, :],
                    'pelvis_delta': pelvis_delta,
                }
            )
            canonicalized_history_primitive_dict, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict, use_predicted_joints=optim_args.use_predicted_joints)
            transf_rotmat, transf_transl = canonicalized_history_primitive_dict['transf_rotmat'], \
            canonicalized_history_primitive_dict['transf_transl']
            history_motion = primitive_utility.dict_to_tensor(blended_feature_dict)
            history_motion = dataset.normalize(history_motion)  # [> B, T, D]

        return motion_sequences, history_motion, transf_rotmat, transf_transl

    optim_steps = optim_args.optim_steps
    lr = optim_args.optim_lr
    noise = torch.randn(num_rollout, batch_size, *denoiser_args.model_args.noise_shape,
                        device=device, dtype=torch.float32)
    # noise = noise.clip(min=-1, max=1)
    noise = noise * optim_args.init_noise_scale
    noise.requires_grad_(True)
    reduction_dims = list(range(1, len(noise.shape)))
    criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)

    optimizer = torch.optim.Adam([noise], lr=lr)
    t_total_start = time.time()
    for i in tqdm(range(optim_steps)):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if optim_args.optim_anneal_lr:
            frac = 1.0 - i / optim_steps
            lrnow = frac * lr
            optimizer.param_groups[0]["lr"] = lrnow

        t_rollout_start = time.time()

        motion_sequences, new_history_motion_tensor, new_transf_rotmat, new_transf_transl = rollout(noise,
                                                                                                    history_motion_tensor,
                                                                                                    transf_rotmat,
                                                                                                    transf_transl)
        times['rollout'] += time.time() - t_rollout_start
        global_joints = motion_sequences['joints']  # [> B, T, 22, 3]
        B, T, _, _ = global_joints.shape
        
        torch.cuda.empty_cache()
        t_sdf_start = time.time()
        sdf_values = calc_vol_sdf(scene_assets, motion_sequences, transf_rotmat, transf_transl, plot=(optim_args.visualize_sdf==1 and (i % 10 == 0)))
        # sdf_values = calc_vol_sdf(scene_assets, motion_sequences, transf_rotmat, transf_transl, plot=(optim_args.visualize_sdf==1 and (i+1 >= optim_steps or i < 1)))
        # TODO: implement meaningful loss function

        # # FOOT-CONTACT_LOSS, don't how to do this with VolSMPL
        # foot_joints_sdf = joints_sdf[:, :, FOOT_JOINTS_IDX]  # [> B, T, 2]
        # loss_floor_contact = (foot_joints_sdf.amin(dim=-1) - optim_args.contact_thresh).clamp(min=0).mean()
        
        if optim_args.floor_contact_loss == 0: #penalize sdf vals if not within threshold from floor
            loss_floor_contact = torch.relu(sdf_values - optim_args.contact_thresh).amin(dim=-1).mean()
        elif optim_args.floor_contact_loss == 1:
            loss_floor_contact = ((global_joints[:, :, FOOT_JOINTS_IDX, 2] - joint_skin_dist.reshape(1,1,22)[:,:,FOOT_JOINTS_IDX]).amin(dim=-1) - scene_assets['floor_height'] - optim_args.contact_thresh).clamp(min=0).mean()

        # # GENERAL COLLISION_LOSS
        # negative_sdf_per_frame = (joints_sdf - joint_skin_dist.reshape(1, 1, 22)).clamp(max=0).sum( # i think joint_skin_dist is the distance to the skin (and no more necessary, thanks VolSMPL)
        #     dim=-1)  # [> B, T], clip negative sdf, sum over joints
        # negative_sdf_mean = negative_sdf_per_frame.mean()
        # loss_collision = -negative_sdf_mean
        loss_collision = torch.relu(-sdf_values).sum(dim=(-2,-1)).mean()

        times['sdf'] += time.time() - t_sdf_start

        # OTHER LOSS VALUES (just leave or?)
        loss_joints = criterion(motion_sequences['joints'][:, -1, joints_mask], goal_joints[:, joints_mask])
        loss_jerk = calc_jerk(motion_sequences['joints'])
        # print("loss_joints shape", loss_joints.shape)
        # print("loss_collision shape", loss_collision.shape)

        # TOTAL LOSS
        # loss = loss_joints + optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk
        # loss = loss_joints + optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk + optim_args.weight_contact * loss_floor_contact
        loss = loss_joints + (i/optim_steps) * optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk + optim_args.weight_contact * loss_floor_contact

        loss.backward()
        if optim_args.optim_unit_grad:
            noise.grad.data /= noise.grad.norm(p=2, dim=reduction_dims, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        # print(f'[{i}/{optim_steps}] loss: {loss.item()} loss_joints: {loss_joints.item()} loss_collision: {loss_collision.item()} loss_jerk: {loss_jerk.item()}')
        print(f'[{i}/{optim_steps}] loss: {loss.item()} loss_joints: {loss_joints.item()} loss_collision: {loss_collision.item()} loss_jerk: {loss_jerk.item()} loss_floor_contact: {loss_floor_contact.item()}')

    times['total'] += time.time() - t_total_start
    times['nbr_iterations'] = optim_steps

    losses['total'] = loss
    losses['joints'] = loss_joints
    losses['collision'] = loss_collision
    losses['jerk'] = loss_jerk
    losses['floor'] = loss_floor_contact

    for key in motion_sequences:
        if torch.is_tensor(motion_sequences[key]):
            motion_sequences[key] = motion_sequences[key].detach()
    motion_sequences['texts'] = texts
    return motion_sequences, new_history_motion_tensor.detach(), new_transf_rotmat.detach(), new_transf_transl.detach()

if __name__ == '__main__':
    optim_args = tyro.cli(OptimArgs)
    # TRY NOT TO MODIFY: seeding
    random.seed(optim_args.seed)
    np.random.seed(optim_args.seed)
    torch.manual_seed(optim_args.seed)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = optim_args.torch_deterministic
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device
    
    denoiser_args, denoiser_model, vae_args, vae_model = load_mld(optim_args.denoiser_checkpoint, device)
    denoiser_checkpoint = Path(optim_args.denoiser_checkpoint)
    save_dir = denoiser_checkpoint.parent / denoiser_checkpoint.name.split('.')[0] / 'optim'
    save_dir.mkdir(parents=True, exist_ok=True)
    optim_args.save_dir = save_dir

    diffusion_args = denoiser_args.diffusion_args
    assert 'ddim' in optim_args.respacing
    diffusion_args.respacing = optim_args.respacing
    print('diffusion_args:', asdict(diffusion_args))
    diffusion = create_gaussian_diffusion(diffusion_args)
    sample_fn = diffusion.ddim_sample_loop_full_chain
    
    print("######## HALLO #######")
    print(vae_args.data_args.cfg_path)
    print(vae_args.data_args.data_dir)

    # load initial seed dataset
    dataset = SinglePrimitiveDataset(cfg_path=vae_args.data_args.cfg_path,  # cfg path from model checkpoint
                                     dataset_path=vae_args.data_args.data_dir,  # dataset path from model checkpoint
                                     sequence_path='./data/stand.pkl',
                                     batch_size=optim_args.batch_size,
                                     device=device,
                                     enforce_gender='male',
                                     enforce_zero_beta=1,
                                     )
    
    future_length = dataset.future_length
    history_length = dataset.history_length
    primitive_length = history_length + future_length
    primitive_utility = dataset.primitive_utility
    batch_size = optim_args.batch_size
   
    with open('./data/joint_skin_dist.json', 'r') as f:
        joint_skin_dist = json.load(f)
        joint_skin_dist = torch.tensor(joint_skin_dist, dtype=torch.float32, device=device)
        joint_skin_dist = joint_skin_dist.clamp(min=optim_args.contact_thresh)  # [22]

    """optimization config"""
    with open(optim_args.interaction_cfg, 'r') as f:
        interaction_cfg = json.load(f)
    interaction_name = interaction_cfg['interaction_name'].replace(' ', '_')
    scene_dir = Path(interaction_cfg['scene_dir'])
    scene_dir = Path(scene_dir)

    # read scene file
    scene_mesh = trimesh.load(scene_dir / interaction_cfg["scene_file"], process=False, force='mesh') # open obj-file as mesh
    scene_points = torch.tensor(scene_mesh.vertices, dtype=torch.float32, device=device) # only points/vertices of obj-mesh
    
    if torch.isnan(scene_points).any() or torch.isinf(scene_points).any():
        raise ValueError("[ERROR] Scene points contain NaN or Inf values.")

    # change axis orientation
    scene_points[:, [0,1,2]] = scene_points[:, [0,2,1]]
    scene_points[:, 1] = -scene_points[:, 1]

    # save infos as scene dict
    scene_assets = {
        'scene_mesh': scene_mesh, # obj-mesh
        'scene_points': scene_points, # only 3D points
        'floor_height': interaction_cfg['floor_height'],
        'nbr_points': scene_points.shape[0],
    }

    # create dict for time measurements
    times = {
        'total': 0.0,
        'rollout': 0.0,
        'sdf': 0.0,
        'nbr_iterations': 0,
    }

    # dict to save losses
    losses = {
        'total': 0.0,
        'joints': 0.0,
        'collision': 0.0,
        'jerk': 0.0,
        'floor': 0.0,
    }

    out_path = optim_args.save_dir
    filename = f'guidance{optim_args.guidance_param}_seed{optim_args.seed}'
    if optim_args.respacing != '':
        filename = f'{optim_args.respacing}_{filename}'
    # if optim_args.smooth:
    #     filename = f'smooth_{filename}'
    if optim_args.zero_noise:
        filename = f'zero_noise_{filename}'
    if optim_args.use_predicted_joints:
        filename = f'use_pred_joints_{filename}'
    filename = f'{interaction_name}_{filename}'
    filename = f'{filename}_contact{optim_args.weight_contact}_thresh{optim_args.contact_thresh}_collision{optim_args.weight_collision}_jerk{optim_args.weight_jerk}'
    out_path = out_path / filename
    out_path.mkdir(parents=True, exist_ok=True)

    batch = dataset.get_batch(batch_size=optim_args.batch_size)
    input_motions, model_kwargs = batch[0]['motion_tensor_normalized'], {'y': batch[0]}
    del model_kwargs['y']['motion_tensor_normalized']
    gender = model_kwargs['y']['gender'][0]
    betas = model_kwargs['y']['betas'][:, :primitive_length, :].to(device)  # [> B, H+F, 10]
    pelvis_delta = primitive_utility.calc_calibrate_offset({
        'betas': betas[:, 0, :],
        'gender': gender,
    })
    # print(input_motions, model_kwargs)
    input_motions = input_motions.to(device)  # [> B, D, 1, T]
    motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)  # [> B, T, D]
    init_history_motion = motion_tensor[:, :history_length, :]  # [> B, H, D]

    print(f"Weights: contact={optim_args.weight_contact}, collision={optim_args.weight_collision}, jerk={optim_args.weight_jerk}")

    all_motion_sequences = None
    for interaction_idx, interaction in enumerate(interaction_cfg['interactions']):
        cache_path = out_path / f'cache_{interaction_idx}.pkl'
        if cache_path.exists() and optim_args.load_cache:
            with open(cache_path, 'rb') as f:
                all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = pickle.load(f)
            tensor_dict_to_device(all_motion_sequences, device)
            history_motion_tensor = history_motion_tensor.to(device)
            transf_rotmat = transf_rotmat.to(device)
            transf_transl = transf_transl.to(device)
        else:
            text_prompt = interaction['text_prompt']
            goal_joints = torch.zeros(batch_size, 22, 3, device=device, dtype=torch.float32)
            goal_joints[:, 0] = torch.tensor(interaction['goal_joints'][0], device=device, dtype=torch.float32)
            joints_mask = torch.zeros(22, device=device, dtype=torch.bool)
            joints_mask[0] = 1

            if interaction_idx == 0:
                history_motion_tensor = init_history_motion
                initial_joints = torch.tensor(interaction['init_joints'], device=device,
                                              dtype=torch.float32)  # [3, 3]
                # initial_joints[:, 2] = -pelvis_feet_height + scene_assets['floor_height']  # snap to floor
                transf_rotmat, transf_transl = get_new_coordinate(initial_joints[None])
                transf_rotmat = transf_rotmat.repeat(batch_size, 1, 1)
                transf_transl = transf_transl.repeat(batch_size, 1, 1)
                # visualize
                # transform = np.eye(4)
                # transform[:3, :3] = transf_rotmat[0].cpu().numpy()
                # transform[:3, 3] = transf_transl[0].cpu().numpy()
                # axis_mesh = trimesh.creation.axis(axis_length=0.1, transform=transform)
                # transform = np.eye(4)
                # transform[:3, 3] = goal_joints[0, 0].cpu().numpy()
                # goal_mesh = trimesh.creation.axis(axis_length=0.1, transform=transform)
                # (axis_mesh + goal_mesh + scene_assets['scene_with_floor_mesh']).show()

            # ADD VolumetricSMPL
            # get batch size
            _,num_rollout = get_text_promts(text_prompt)
            batch_size_vol_ = (future_length * num_rollout + history_length) * batch_size
            print(f"Batch sizes: {batch_size_vol_} = ({future_length}*{num_rollout} + {history_length}) * {batch_size}")
            # make (high-level) SMPLX body model
            vol_body_model_dict = {
                'male':   smplx.create(model_path=body_model_dir, model_type='smplx', batch_size=batch_size_vol_, gender='male',   num_betas=10, ext='npz', num_pca_comps=12),
                'female': smplx.create(model_path=body_model_dir, model_type='smplx', batch_size=batch_size_vol_, gender='female', num_betas=10, ext='npz', num_pca_comps=12)
            }
            # attach volumetric representation
            attach_volume(vol_body_model_dict['male'])
            attach_volume(vol_body_model_dict['female'])

            print("""# ------------------------ #\n[Start] Optimizing Routine""")
            motion_sequences, history_motion_tensor, transf_rotmat, transf_transl = optimize(
                history_motion_tensor, transf_rotmat, transf_transl, text_prompt, goal_joints, joints_mask)

            if all_motion_sequences is None:
                all_motion_sequences = motion_sequences
                all_motion_sequences['goal_location_list'] = [goal_joints[0, 0].cpu()]
                num_frames = all_motion_sequences['joints'].shape[1]
                all_motion_sequences['goal_location_idx'] = [0] * num_frames
            else:
                for key in motion_sequences:
                    if torch.is_tensor(motion_sequences[key]):
                        # print(key, all_motion_sequences[key].shape, motion_sequences[key].shape)
                        all_motion_sequences[key] = torch.cat([all_motion_sequences[key], motion_sequences[key]], dim=1)
                all_motion_sequences['texts'] += motion_sequences['texts']
                all_motion_sequences['goal_location_list'] += [goal_joints[0, 0].cpu()]
                num_goals = len(all_motion_sequences['goal_location_list'])
                num_frames = all_motion_sequences['joints'].shape[1]
                all_motion_sequences['goal_location_idx'] += [num_goals - 1] * num_frames
            with open(cache_path, 'wb') as f:
                pickle.dump([all_motion_sequences, history_motion_tensor, transf_rotmat, transf_transl], f)
            print("[End] Optimizing Routine\n# ------------------------ #")

    for idx in range(batch_size):
        sequence = {
            'texts': all_motion_sequences['texts'],
            'scene_path': scene_dir / 'scene_with_floor.obj',
            'goal_location_list': all_motion_sequences['goal_location_list'],
            'goal_location_idx': all_motion_sequences['goal_location_idx'],
            'gender': all_motion_sequences['gender'],
            'betas': all_motion_sequences['betas'][idx],
            'transl': all_motion_sequences['transl'][idx],
            'global_orient': all_motion_sequences['global_orient'][idx],
            'body_pose': all_motion_sequences['body_pose'][idx],
            'joints': all_motion_sequences['joints'][idx],
            'history_length': history_length,
            'future_length': future_length,
        }
        tensor_dict_to_device(sequence, 'cpu')
        with open(out_path / f'volSPML_sample_{idx}.pkl', 'wb') as f:
            pickle.dump(sequence, f)

        # export smplx sequences for blender
        if optim_args.export_smpl:
            poses = transforms.matrix_to_axis_angle(
                torch.cat([sequence['global_orient'].reshape(-1, 1, 3, 3), sequence['body_pose']], dim=1)
            ).reshape(-1, 22 * 3)
            poses = torch.cat([poses, torch.zeros(poses.shape[0], 99).to(dtype=poses.dtype, device=poses.device)],
                              dim=1)
            data_dict = {
                'mocap_framerate': dataset.target_fps,  # 30
                'gender': sequence['gender'],
                'betas': sequence['betas'][0, :10].detach().cpu().numpy(),
                'poses': poses.detach().cpu().numpy(),
                'trans': sequence['transl'].detach().cpu().numpy(),
            }
            with open(out_path / f'volSMPL_sample_{idx}_smplx.npz', 'wb') as f:
                np.savez(f, **data_dict)

    print(f'[Done] Results are at [{out_path.absolute()}]')

    print(f"""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                Optimization Results                        ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║ Times:                                                                     ║
    ║     total   = {times['total']:<10.5f}                                                   ║
    ║     rollout = {times['rollout']:<10.5f}                                                   ║
    ║     sdf     = {times['sdf']:<10.5f}                                                   ║
    ║     nbr it. = {times['nbr_iterations']:<10}                                                   ║
    ║                                                                            ║
    ║ Losses:                                                                    ║
    ║     total     = {losses['total']:<15.8f}                                            ║
    ║     joints    = {losses['joints']:<15.8f}                                            ║
    ║     collision = {losses['collision']:<15.8f}                                            ║
    ║     jerk      = {losses['jerk']:<15.8f}                                            ║
    ║     floor     = {losses['floor']:<15.8f}                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
