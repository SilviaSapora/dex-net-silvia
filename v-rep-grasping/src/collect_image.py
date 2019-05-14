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

def load_mesh(mesh_path):
    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    mesh = trimesh.load_mesh(mesh_path)
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh

class CollectImage(object):

    def __init__(self, mesh_path):
        # Use the spawn_headless = False / True flag to view with GUI or not
        spawn_params = {'port': 19997,
            'ip': '127.0.0.1',
            'vrep_path': '/home/silvia/Downloads/V-REP_PRO_EDU_V3_5_0_Linux/vrep',
            'scene_path': None,
            'exit_on_stop': True,
            'spawn_headless': False,
            'spawn_new_console': True}

        self.sim = SI.SimulatorInterface(**spawn_params)
        self.mesh_path = mesh_path

    def load_new_object(self):
        mesh = load_mesh(self.mesh_path)

        mass = mesh.mass_properties['mass'] * 10
        com = mesh.mass_properties['center_mass']
        inertia = mesh.mass_properties['inertia'] * 5

        self.sim.load_object(self.mesh_path, com, mass, inertia.flatten())

    def drop_object(self, stable_pose):
        self.sim.run_threaded_drop(stable_pose)
        self.sim.set_object_pose(stable_pose[:3].flatten(), True)

    def collect_image(self, camera_pose, im_height, im_width):
        # print(camera_pose)
        self.sim.set_camera_pose_from_obj_pose(camera_pose)
        self.sim.set_camera_resolution(im_height, im_width)
        rgb_image, depth_image = self.sim.camera_images()
        
        depth_image = np.float32(depth_image.reshape([1,im_height,im_width,1]))
        rgb_image = np.uint8(rgb_image.reshape([1,im_height,im_width,3]))
        
        return rgb_image, depth_image

    def check_collision(self, gripper_pose=None):
        collision = self.sim.set_gripper_pose(gripper_pose)
        return collision

    def stop(self):
        self.sim.stop()
        return

#if __name__ == '__main__':
    # check_collision(sim)
