import os
import sys
import trimesh
import mesh2sdf
import numpy as np
import time
import skimage
import json
from pathlib import Path
from tqdm import tqdm

def test_mesh2sdf(filename, size=128, level=1/128, mesh_scale = 0.8):
    # level = 2 / size  # recommended level = 2 / size
    mesh = trimesh.load(filename, force='mesh')

    # set floor
    vertices = mesh.vertices
    floor = np.min(vertices[:,1])
    vertices = vertices + np.array([0,-floor,0])

    # change axis orientation
    vertices[:, [0,1,2]] = vertices[:, [2,0,1]]
    vertices[:, 0] = -vertices[:, 0]

    # normalize mesh
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # fix mesh
    # t0 = time.time()
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    # t1 = time.time()

    # output
    mesh.vertices = mesh.vertices / scale + center
    mesh.export(filename[:-4] + '_sdf.obj')
    np.save(filename[:-4] + '_sdf.npy', sdf)
    with open(filename[:-4] + '_sdf.json', 'w') as f:
        json.dump({
            'center': center.tolist(),
            'scale': scale,
            'level': level,
            'size': size,
        }, f)
    # print('It takes %.4f seconds to process %s' % (t1-t0, filename))


scene_dir_list = [Path('/home/sebastian/GIT/DH/iDART/data/scenes/cab_e'),
                  Path('/home/sebastian/GIT/DH/iDART/data/scenes/cab_g_benches'),
                  Path('/home/sebastian/GIT/DH/iDART/data/scenes/foodlab_0312'),
                  Path('/home/sebastian/GIT/DH/iDART/data/scenes/cnb_dlab_0215'),
                  Path('/home/sebastian/GIT/DH/iDART/data/scenes/seminar_h53_0218')]
# scene_dir_list = [Path('/home/kaizhao/projects/multiskill/data/scenes/demo/62b1714ef66f4e0d9f42dcd12efb3f52/')]
size = 256
for scene_dir in tqdm(scene_dir_list):
    # filename = str(scene_dir / 'scene.obj')
    filename = str(scene_dir) + '/' + str(str(scene_dir).split('/')[-1] + '.obj')
    test_mesh2sdf(filename, size=size, level=1/size, mesh_scale=0.8)