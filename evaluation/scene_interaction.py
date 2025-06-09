#!/usr/bin/env python3
"""
EVALUATION OF OPTIMIZATION WITH VolumetricSMPL
Course Group Project - Digital Humans, ETH Zürich, Spring 2025

Description: 
    This script evaluates the optimization of a scene interaction using the VolumetricSMPL model.
    It opens the motion sequence from a pickle file, calculates the signed distance function (SDF) values for the scene points,
    and computes various losses such as joint loss, collision loss, jerk loss, and floor contact loss.

Filename: scene_interaction.py
Author:   Sebastian Heckers
Date:     2025-06-09
Version:  1.0
Based on: mld/optim_scene_mld.py
"""

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
    smpl_file: str = './mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/optim/seminar_h53_0218_walk_test_8000_use_pred_joints_ddim10_guidance5.0_seed0_contact0.1_thresh0.0_collision1.0_jerk0.1/msh2sdf_sample_0.pkl'

    visualize_sdf: int = 0


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
    
    # repeat points for each time step
    stacked_pointspoints = all_points.unsqueeze(0).repeat(B, T, 1, 1)  # Shape: [T, N, 3]

    # Generate subset time frames indices
    if rand_idxs:
        indices = torch.randint(0, T, (T//s,))
    else:
        indices = torch.arange(0, T, step=s)
    
    t_ = len(indices)

    # select subset of time steps
    points = stacked_pointspoints[:,indices,:,:]  # Shape: [T//s, N, 3]
    # points = points.permute(2,0,1,3)
    points = points.reshape(B*t_, -1, 3)  # Shape: [B * T//s, N, 3]

    return points, indices

def plot_sampled_points_with_sdf(points, sdf_values, smpl_output, T=98, title="Sampled Points with SDF Values"):
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


    print("[Start] Plot scatter SDF-points")
    for t_ in tqdm(range(T)):
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with SDF values as colors
        scatter = ax.scatter(points_np[0,:, 0], points_np[0,:, 1], points_np[0,:, 2], c=sdf_values_np[t_,:], cmap='viridis', s=5, alpha=0.4)

        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("SDF Values")

        # scatter plot of SMPL body
        ax.scatter(smpl_output_np[t_,:,0],smpl_output_np[t_,:,1],smpl_output_np[t_,:,2],c='black', s=10, alpha=1.0)

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
        plt.savefig(f"bin/imgs/sdf-points-eval-{t_}.png")
        plt.close()

    print(f'[Done] Plotting SDF-Points [bin/imgs/sdf-points-eval-{T}.png]')

def calc_vol_sdf(scene_assets, motion_sequences,plot=False):
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
    T, J, _ = motion_sequences['joints'].shape
    # print('B, T, J:', B, T, J)

    # get body pose parameterscon
    betas = motion_sequences['betas'].reshape(T, 10).to(device=device) #body shape coeffs, don't rotate! 
    # > (B, T, 3, 3) -> (B, 3)
    global_orient = motion_sequences['global_orient'].to(device=device) #global rotation of root joint
    #  (B, T, 21, 3, 3)-> (B, (J*3))
    body_pose = motion_sequences['body_pose'].to(device=device) #relative rotations of other joints
    # (B, T, 3) -> (B, 3)
    transl = motion_sequences['transl'].to(device=device) #global translation of the root
    
    #convert back to axis-angle
    global_orient = matrix_to_axis_angle(global_orient) 
    body_pose = matrix_to_axis_angle(body_pose)

    #reshape for required input size for SMPL-X
    global_orient = global_orient.reshape(T, 3).to(device=device)
    body_pose = body_pose.reshape(T, 21, 3).to(device=device)
    transl = transl.reshape(T, 3).to(device=device)

    # get smplx model
    body_model = vol_body_model_dict[gender].to(device=device)

    # get "current" smpl body model
    smpl_output = body_model(betas=betas,
                             global_orient=global_orient,
                             body_pose=body_pose,
                             transl=transl,
                             return_verts=True, return_full_pose=True)

    # get scene points
    scene_points, idxs = sample_scene_points(scene_assets,1,1,s=1)
    # scene_points = scene_points.to(device=device)  # shape: [t, N, 3]

    # get SMPL body_model parameters
    smpl_output_full_pose = smpl_output.full_pose.reshape(T,-1,3)
    smpl_output_vertices = smpl_output.vertices.reshape(T,-1,3)
    smpl_output_joints = smpl_output.joints.reshape(T,-1,3)
    
    # query sampled points for each batch
    sdf_values = torch.zeros(T, scene_assets['nbr_points'], device=device)
    print("[Start] Query SDF-points")
    for i_ in tqdm(range(T)):
        torch.cuda.empty_cache()
        # select specific smpl_output for each batch
        subset_smpl_output = smpl_output
        subset_smpl_output.full_pose = smpl_output_full_pose[i_, ...].reshape(1,-1,3)
        subset_smpl_output.vertices =  smpl_output_vertices[i_, ...].reshape(1,-1,3)
        subset_smpl_output.joints =    smpl_output_joints[i_, ...].reshape(1,-1,3)
        # query sampled points for sdf values
        sdf_values[i_, :] = body_model.volume.query_fast(scene_points, subset_smpl_output)
    print(f"[Done] Query SDF-points [{sdf_values.shape}]")

    if plot:
        plot_sampled_points_with_sdf(scene_points, sdf_values, smpl_output_vertices, T=T)

    return sdf_values # shape: [B, t_, N]

def calc_jerk(joints):
    vel = joints[1:] - joints[:-1]  # --> > B x T-1 x 22 x 3
    acc = vel[1:] - vel[:-1]  # --> > B x T-2 x 22 x 3
    jerk = acc[1:] - acc[:-1]  # --> > B x T-3 x 22 x 3
    jerk = torch.sqrt((jerk ** 2).sum(dim=-1))  # --> > B x T-3 x 22, compute L1 norm of jerk
    jerk = jerk.amax(dim=[1])  # --> > B, Get the max of the jerk across all joints and frames

    return jerk

def load_motion_sequence(pickle_file_path):
    """
    Loads and reads a motion sequence from a pickle file.

    Args:
        pickle_file_path (str or Path): Path to the pickle file.
    Returns:
        dict: The motion sequence data.
    """
    try:
        with open(pickle_file_path, 'rb') as f:
            motion_sequence = pickle.load(f)
        print(f"[INFO] Successfully loaded motion sequence from {pickle_file_path}")
        return motion_sequence
    except FileNotFoundError:
        print(f"[ERROR] File not found: {pickle_file_path}")
    except pickle.UnpicklingError:
        print(f"[ERROR] Failed to unpickle the file: {pickle_file_path}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == '__main__':
    optim_args = tyro.cli(OptimArgs)
    
    device = torch.device(optim_args.device if torch.cuda.is_available() else "cpu")
    optim_args.device = device

    batch_size = optim_args.batch_size

    pickle_file_path = optim_args.smpl_file
    # pickle_file_path = "./bin/output/sample_0.pkl"
    motion_sequence = load_motion_sequence(pickle_file_path)

    gender = motion_sequence['gender']
    T, J, _ = motion_sequence['joints'].shape

    # ADD VolumetricSMPL
    # make (high-level) SMPLX body model
    vol_body_model_dict = {
        'male':   smplx.create(model_path=body_model_dir, model_type='smplx', batch_size=T, gender='male',   num_betas=10, ext='npz', num_pca_comps=12),
        'female': smplx.create(model_path=body_model_dir, model_type='smplx', batch_size=T, gender='female', num_betas=10, ext='npz', num_pca_comps=12)
    }
    # attach volumetric representation
    attach_volume(vol_body_model_dict['male'])
    attach_volume(vol_body_model_dict['female'])

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

    # dict to save losses
    losses = {
        'total': 0.0,
        'joints': 0.0,
        'collision': 0.0,
        'jerk': 0.0,
        'floor': 0.0,
    }

    criterion = torch.nn.HuberLoss(reduction='none', delta=1.0)
    joints_mask = torch.zeros(22, dtype=torch.bool)
    joints_mask[0] = 1
    goal_joints = motion_sequence['goal_location_list'][0]

    with torch.no_grad():
        sdf_values = calc_vol_sdf(scene_assets, motion_sequence, plot=(optim_args.visualize_sdf==1))
        # FOOT-CONTACT_LOSS, don't how to do this with VolSMPL
        if optim_args.floor_contact_loss == 0: 
            loss_floor_contact = torch.relu(sdf_values - optim_args.contact_thresh).amin(dim=-1).to(device=device)
        elif optim_args.floor_contact_loss == 1:
            loss_floor = ((global_joints[:, :, FOOT_JOINTS_IDX, 2] - joint_skin_dist.reshape(1,1,22)[FOOT_JOINTS_IDX]).amin(dim=-1) - optim_args.contact_thresh).clamp(min=0)

        print(sdf_values.shape)
        # GENERAL COLLISION_LOSS
        loss_collision = torch.relu(-sdf_values).sum(-1)
        # loss_collision = torch.relu(-sdf_values).sum(dim=(-2,-1)).mean()

        # OTHER LOSS VALUES (just leave or?)
        loss_joints = criterion(motion_sequence['joints'][:, joints_mask], goal_joints.unsqueeze(0).repeat(T, 1, 1)).norm(dim=-1)
        loss_jerk = torch.zeros(T)
        loss_jerk[3:] = calc_jerk(motion_sequence['joints'])

        # print(f'[{i}/{optim_steps}] loss: {loss.item()} loss_joints: {loss_joints.item()} loss_collision: {loss_collision.item()} loss_jerk: {loss_jerk.item()}')
        for i_ in range(0,T,10):
            print(f'[{i_}/{T}] loss joints: {loss_joints[i_].item():<12.10} collision: {loss_collision[i_].item():<12.10} jerk: {loss_jerk[i_].item():<12.10} floor_contact: {loss_floor_contact[i_].item():<12.10}')

        # TOTAL LOSS
        # loss = loss_joints + optim_args.weight_collision * loss_collision + optim_args.weight_jerk * loss_jerk
        loss = loss_joints[-1].mean() + optim_args.weight_collision * loss_collision.sum() + optim_args.weight_jerk * loss_jerk[3:].mean() + optim_args.weight_contact * loss_floor_contact.mean()

    print(f'[Total] loss: {loss.item()} loss_joints: {loss_joints[-1].mean().item()} loss_collision: {loss_collision.mean().item()} loss_jerk: {loss_jerk[3:].mean().item()} floor_contact: {loss_floor_contact.mean().item()}')
    losses['total'] = loss.item()
    losses['joints'] = loss_joints[-1].mean().item()
    losses['collision'] = loss_collision.sum().item()
    losses['jerk'] = loss_jerk[3:].mean().item()
    losses['floor'] = loss_floor_contact.mean().item()

    log_file_path = optim_args.save_dir + "/losses_log.txt"

    # Prepare the line to append
    losses_line = (
        f"{interaction_name}_"
        + ("VolSMPL" if "vol" in optim_args.smpl_file else "Msh2SDF") + ", "
        f"{losses['total']:.8f}, "
        f"{losses['joints']:.8f}, "
        f"{losses['collision']:.8f}, "
        f"{losses['jerk']:.8f}, "
        f"{losses['floor']:.8f}\n"
    )

    # Append the line to the file
    with open(log_file_path, "a") as log_file:
        log_file.write(losses_line)

    print(f"[INFO] Losses logged to {log_file_path}")
    print(f"""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                          Optimization Results                              ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║ Interaction Name: {interaction_name:<50}       ║
    ║ SMPL Path: {optim_args.smpl_file[:60]:<60}... ║
    ║            {optim_args.smpl_file[60:120]:<60}... ║
    ║            {optim_args.smpl_file[120:180]:<60}... ║
    ║            {optim_args.smpl_file[180:210]:<60}    ║
    ║ Configuration: {optim_args.interaction_cfg:<60}║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║ Optimization Parameters:                                                   ║
    ║     Guidance Param   = {optim_args.guidance_param:<10.5f}                                          ║
    ║     Respacing        = {optim_args.respacing:<10}                                          ║
    ║     Batch Size       = {optim_args.batch_size:<10}                                          ║
    ║     Optim LR         = {optim_args.optim_lr:<10.5f}                                          ║
    ║     Optim Steps      = {optim_args.optim_steps:<10}                                          ║
    ║     Weight Jerk      = {optim_args.weight_jerk:<10.5f}                                          ║
    ║     Weight Collision = {optim_args.weight_collision:<10.5f}                                          ║
    ║     Weight Contact   = {optim_args.weight_contact:<10.5f}                                          ║
    ║     Contact Thresh   = {optim_args.contact_thresh:<10.5f}                                          ║
    ║     Init Noise Scale = {optim_args.init_noise_scale:<10.5f}                                          ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║ Losses:                                                                    ║
    ║     Total Loss      = {losses['total']:<15.8f}                                      ║
    ║     Joints Loss     = {losses['joints']:<15.8f}                                      ║
    ║     Collision Loss  = {losses['collision']:<15.8f}                                      ║
    ║     Jerk Loss       = {losses['jerk']:<15.8f}                                      ║
    ║     Floor Loss      = {losses['floor']:<15.8f}                                      ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
