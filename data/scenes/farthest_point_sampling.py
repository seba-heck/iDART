"""
Script for sampling 2000, 4000, 6000, and 8000 points from object meshes
using the `fpsample` library. The script processes a predefined list of 
objects, either with or without vertex color information, and performs 
farthest point sampling (FPS) on their geometry.

Input Format:
    - Objects with color:       v x y z r g b
    - Objects without color:    v x y z

Output Files (per object):
    - object_2000.obj
    - object_4000.obj
    - object_6000.obj
    - object_8000.obj

Each output file contains the sampled vertices, preserving original color if 
available.

--------------------------------------------------------------------------------
Author: David Blickenstorfer (assisted by GPT-4o)
Created: 13/05/2025
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
objects = ['cab_e','cab_g_benches', 'seminar_d78','seminar_j716', 'cab_h_tables', 'kitchen_gfloor', 
           'seminar_g110', 'seminar_h52']

objects_without_color = ['cnb_dlab_0215', 'seminar_g110_0315', 'seminar_h53_0218', 'cnb_dlab_0225',
                         'foodlab_0312', 'seminar_d78_0318', 'seminar_g110_0415',]

# iterate over all object in objects
for obj in objects:
    print(f"process {obj}")
    print("import object data")
    points, colors = read_obj_with_color(f"{obj}/{obj}.obj")

    # Vanilla FPS
    print("farthest point sampling")
    fps_indices_2000 = fpsample.fps_sampling(points, 2000)
    fps_indices_4000 = fpsample.fps_sampling(points, 4000)
    fps_indices_6000 = fpsample.fps_sampling(points, 6000)
    fps_indices_8000 = fpsample.fps_sampling(points, 8000)

    # Get actual sampled points
    fps_points_2000 = points[fps_indices_2000]
    fps_colors_2000 = colors[fps_indices_2000]

    fps_points_4000 = points[fps_indices_4000]
    fps_colors_4000 = colors[fps_indices_4000]

    fps_points_6000 = points[fps_indices_6000]
    fps_colors_6000 = colors[fps_indices_6000]

    fps_points_8000 = points[fps_indices_8000]
    fps_colors_8000 = colors[fps_indices_8000]

    # store cab_e as reduced point cloud data
    print("export fps object file")
    write_obj_with_color(f"{obj}/{obj}_2000.obj", fps_points_2000, fps_colors_2000)
    write_obj_with_color(f"{obj}/{obj}_4000.obj", fps_points_4000, fps_colors_4000)
    write_obj_with_color(f"{obj}/{obj}_6000.obj", fps_points_6000, fps_colors_6000)
    write_obj_with_color(f"{obj}/{obj}_8000.obj", fps_points_8000, fps_colors_8000)
    

# iterate over all object in objects
for obj in objects_without_color:
    print(f"process {obj}")
    print("import object data")
    points = read_obj(f"{obj}/{obj}.obj")

    # Vanilla FPS
    print("farthest point sampling")
    fps_indices_2000 = fpsample.fps_sampling(points, 2000)
    fps_indices_4000 = fpsample.fps_sampling(points, 4000)
    fps_indices_6000 = fpsample.fps_sampling(points, 6000)
    fps_indices_8000 = fpsample.fps_sampling(points, 8000)

    # Get actual sampled points
    fps_points_2000 = points[fps_indices_2000]
    fps_points_4000 = points[fps_indices_4000]
    fps_points_6000 = points[fps_indices_6000]
    fps_points_8000 = points[fps_indices_8000]

    # store cab_e as reduced point cloud data
    print("export fps object file")
    write_obj(f"{obj}/{obj}_2000.obj", fps_points_2000)
    write_obj(f"{obj}/{obj}_4000.obj", fps_points_4000)
    write_obj(f"{obj}/{obj}_6000.obj", fps_points_6000)
    write_obj(f"{obj}/{obj}_8000.obj", fps_points_8000)
