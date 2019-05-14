import os
import sys
import glob
import h5py
import numpy as np
import trimesh
import time
import math
from scipy import misc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, '/home/silvia/dex-net/v-rep-grasping/')

import lib
import lib.utils
from lib.config import config_mesh_dir, config_output_collected_dir
import vrep
vrep.simxFinish(-1)
import simulator as SI

sys.path.insert(0, '/home/silvia/dex-net/')
#from stable_pose_grasps import antipodal_grasp_sampler
from store_grasps import SDatabase

def load_mesh(mesh_path):
    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    mesh = trimesh.load_mesh(mesh_path)
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh

def generate_shapes(sim):
    for i in range(0,1):
        object_name = 'procedural_obj_v2_' + str(i)
        mesh_path = '/home/silvia/dex-net/generated_shapes/' + object_name + '.obj'
        if os.path.isfile(mesh_path):
            continue
        sim.create_object(mesh_path)
    # CREATE MESH
    # mesh = load_mesh(mesh_path)
    # mass = mesh.mass_properties['mass'] * 10
    # com = mesh.mass_properties['center_mass']
    # inertia = mesh.mass_properties['inertia'] * 5
    # sim.load_object(mesh_path, com, mass, inertia.flatten())
    #db.database_save(object_name, mesh_path)



if __name__ == '__main__':

    # Use the spawn_headless = False / True flag to view with GUI or not
    spawn_params = {'port': 19997,
                    'ip': '127.0.0.1',
                    'vrep_path': None,
                    'scene_path': None,
                    'exit_on_stop': True,
                    'spawn_headless': False,
                    'spawn_new_console': True}

    # Sample way for calling VREP on windows by specifying full path:
    # spawn_params['vrep_path'] = 'C:\\Program Files\\V-REP3\\V-REP_PRO_EDU\\vrep.exe'
    
    sim = SI.SimulatorInterface(**spawn_params)
    #sim = []

    if len(sys.argv) == 1:
        generate_shapes(sim)

    else:
        spawn_params['port'] = int(sys.argv[1])

        # List of meshes we should run are stored in a file,
        mesh_list_file = sys.argv[2]
        generate_shapes(sim)
