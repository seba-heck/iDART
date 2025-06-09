#!/usr/bin/env python3
"""
TESTING Navmesh AND Pathfinder
Course Group Project - Digital Humans, ETH Zürich, Spring 2025

Description: 
    This script tests the Navmesh baking functionality using the PathFinder library.

Filename: navmesh.py
Author:   Sebastian Heckers
Date:     2025-06-09
Version:  0.0 (in progress)
"""

from pathfinder import navmesh_baker as nmb
import trimesh
import pickle
import pathfinder as pf

inp_path = "./data/scenes/cab_g_benches/cab_g_benches.obj"
out_path = "./data/scenes/cab_g_benches/cab_g_benches.nav"

def make_navmesh(inp_path,out_path):
    # read scene file
    scene_mesh = trimesh.load(inp_path, force='mesh') # open obj-file as mesh

    scene_mesh.vertices[:, [0,1,2]] = scene_mesh.vertices[:, [2,0,1]]
    scene_mesh.vertices[:, 1] = -scene_mesh.vertices[:, 1]

    # Validate and repair the mesh
    if not scene_mesh.is_watertight:
        print("[WARNING] Mesh is not watertight. Attempting to repair...")
        scene_mesh.fix_normals()
        scene_mesh.remove_degenerate_faces()
        scene_mesh.remove_duplicate_faces()
        scene_mesh.remove_infinite_values()

    print(scene_mesh.vertices, scene_mesh.faces)

    # create baker object
    baker = nmb.NavmeshBaker()

    # add geometry, for example a simple plane
    # the first array contains vertex positions, the second array contains polygons of the geometry
    baker.add_geometry(scene_mesh.vertices, scene_mesh.faces)

    # bake navigation mesh
    baker.bake(cell_size=0.1)

    # obtain polygonal description of the mesh
    vertices, polygons = baker.get_polygonization()

    print(vertices)
    print(polygons)

    # Validate the navigation mesh
    if len(vertices) == 0 or len(polygons) == 0:
        raise ValueError("[ERROR] Loaded navigation mesh is empty. Please check the baking process.")

