"""
Script for sampling S points from object meshes using the `fpsample` library. 
The script processes a predefined list of objects, either with or without vertex 
color information, and list of sampling points. It performs farthest point 
sampling (FPS) on their geometry and set the scenes on the floor min(y) = 0.

Remark: Blender's use following format (x, y, z).obj -> (x, up, forward).blend

Input Format:
    - Objects with color:       v x y z r g b
    - Objects without color:    v x y z
    - Sampling points:          [2000, 4000, 6000, 8000]

Output Files (per object):
    - Objects with reduced numbers of vertices and on the floor

Each output file contains the sampled vertices, preserving original color if 
available.

--------------------------------------------------------------------------------
Author: David Blickenstorfer (assisted by GPT-4o)
Created : 13/05/2025 (Script performed FPS)
Modified: 14/05/2025 (Script placed scenes on the floor)
"""

import fpsample
import open3d as o3d
import numpy as np

def read_obj(filepath):
    """
    Read a .obj file and extract vertex positions only.

    Parameters:
        filepath (str): Path to the .obj file.

    Returns:
        np.ndarray: Array of shape (N, 3) containing vertex coordinates.
    """
    positions = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = map(float, parts[1:4])
                    positions.append([x, y, z])
    return np.array(positions)


def write_obj(filename, points):
    """
    Write vertex positions to a .obj file.

    Parameters:
        filename (str): Path to the output .obj file.
        points (np.ndarray): Array of shape (N, 3) with XYZ coordinates.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        for x, y, z in points:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")


def read_obj_with_color(filepath):
    """
    Read a Wavefront .obj file and extract vertex positions and colors.

    The .obj file is expected to have vertex lines in the format:
        v x y z r g b

    Parameters:
        filepath (str): Path to the .obj file.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - positions: Array of shape (N, 3) containing vertex coordinates.
            - colors: Array of shape (N, 3) containing RGB values in [0.0, 1.0].
    """
    positions = []
    colors = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    positions.append([x, y, z])
                    colors.append([r, g, b])
    return np.array(positions), np.array(colors)


def write_obj_with_color(filename, points, colors):
    """
    Write vertex positions and colors to a .obj file using extended vertex format.

    Each vertex line in the output file will be written as:
        v x y z r g b

    Parameters:
        filename (str): Path to the output .obj file.
        points (np.ndarray): Array of shape (N, 3) with XYZ coordinates.
        colors (np.ndarray): Array of shape (N, 3) with RGB values in [0.0, 1.0].

    Returns:
        None
    """ 
    with open(filename, 'w') as f:
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

# list all object files/folders
objects_with_color = ['cab_e','cab_g_benches', 'seminar_d78','seminar_j716', 'cab_h_tables', 'kitchen_gfloor', 
                      'seminar_g110', 'seminar_h52']

objects_without_color = ['cnb_dlab_0215', 'seminar_g110_0315', 'seminar_h53_0218', 'cnb_dlab_0225',
                         'foodlab_0312', 'seminar_d78_0318', 'seminar_g110_0415']

samplingPoints = [2000, 4000, 6000, 8000]

# iterate over all objects with color values
for obj in objects_with_color:        
    print(f"import {obj} and process object data")
    points, colors = read_obj_with_color(f"{obj}/{obj}.obj")

    # Get minimal Up-values (y) and set scene on the floor min(y) = 0
    y_min = points[:, 1].min()
    points[:, 1] -= y_min

    for nSamples in samplingPoints:
        # Vanilla FPS
        print(f"farthest point sampling for S={nSamples}")
        fps_indices = fpsample.fps_sampling(points, nSamples)
        fps_points = points[fps_indices]
        fps_colors = colors[fps_indices]

        # Get actual sampled points
        print(f"export fps object file for S={nSamples}")
        write_obj_with_color(f"{obj}/{obj}_{nSamples}.obj", fps_points, fps_colors)

# iterate over all object in objects
for obj in objects_without_color:        
    print(f"import {obj} and process object data")
    points = read_obj(f"{obj}/{obj}.obj")

    # Get minimal Up-values (y) and set scene on the floor min(y) = 0
    y_min = points[:, 1].min()
    points[:, 1] -= y_min

    for nSamples in samplingPoints:
        # Vanilla FPS
        print(f"farthest point sampling for S={nSamples}")
        fps_indices = fpsample.fps_sampling(points, nSamples)
        fps_points = points[fps_indices]

        # Get actual sampled points
        print(f"export fps object file for S={nSamples}")
        write_obj(f"{obj}/{obj}_{nSamples}.obj", fps_points)