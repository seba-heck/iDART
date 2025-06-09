#!/usr/bin/env python3
"""
MAKE VOLUMETRIC POINTCLOUD
Course Group Project - Digital Humans, ETH Zürich, Spring 2025

Description: 
    This script generates a volumetric point cloud and a surface point cloud, which can be used for testing the VolumetricSMPL model.
    The points are sampled from a cylindrical volume and a cylindrical surface.

Filename: make_obstacle.py
Author:   Sebastian Heckers
Date:     2025-06-09
Version:  1.0
"""

import numpy as np
import argparse

out_path = "tmp.obj"

def sample_point_vol(n=1,h_size=(0,2),r_size=(0,0.3),a_size=(0,2*3.1415),pos=(0,0,0)):
    """Sample points in a cylindrical volume."""
    h = np.random.rand(n)* (h_size[1] - h_size[0]) + h_size[0]
    r = np.random.rand(n)* (r_size[1] - r_size[0]) + r_size[0]
    a = np.random.rand(n)* (a_size[1] - a_size[0]) + a_size[0]

    x = r * np.cos(a)
    y = r * np.sin(a)

    out = np.zeros((n,3))
    out[:,0] = x + pos[0]
    out[:,1] = y + pos[1]
    out[:,2] = h + pos[2]
    return out.reshape(n,3)

def sample_point_suf(n=1,h_size=(0,2),r=0.3,a_size=(0,2*3.1415),pos=(0,0,0)):
    """Sample points on the surface of a cylinder."""
    h = np.random.rand(n)* (h_size[1] - h_size[0]) + h_size[0]
    a = np.random.rand(n)* (a_size[1] - a_size[0]) + a_size[0]

    x = r * np.cos(a)
    y = r * np.sin(a)

    out = np.zeros((n,3))
    out[:,0] = x + pos[0]
    out[:,1] = y + pos[1]
    out[:,2] = h + pos[2]
    return out.reshape(n,3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000, help='Number of points in the volumn.')
    parser.add_argument('--m', type=int, default=1000, help='Number of points on the surface.')
    parser.add_argument('--save', type=bool, default=False, help="Print to terminal or save it as 'tmp.obj' file.")
    args = parser.parse_args()

    # sample points on the surface and in the volume
    points_vol = sample_point_vol(args.n, pos=(0,-1.5,0))
    points_suf = sample_point_suf(args.m, pos=(0,-1.5,0))

    # concatenate and formate the points
    points = np.concatenate((points_vol, points_suf), axis=0)

    points[:, [0,1,2]] = points[:, [0,2,1]]
    points[:, 2] = -points[:, 2]

    # create output string
    output = ""
    for [x,y,z] in points:
        output += f"v {x} {y} {z}\n"

    if args.save:
        with open(out_path, 'w') as fout:
            fout.write(output)
    else:
        print(output)