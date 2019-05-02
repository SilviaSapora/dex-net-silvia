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

from store_grasps_old_db import SDatabase


import gqcnn
import gqcnn.policy_vrep as policy

MESH_PATH = "/home/silvia/dex-net/mesh.obj"
IM_HEIGHT = 480
IM_WIDTH = 640

def load_mesh(mesh_path):
    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    mesh = trimesh.load_mesh(mesh_path)
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh

def save_images(rgb_image, depth_image, postfix, save_dir):
    filename_jpg = os.path.join(save_dir, postfix + '.jpg')

    misc.imsave(filename_jpg, np.uint8(rgb_image))

    im = Image.open(filename_jpg)
    filename_png = os.path.join(save_dir, postfix + '.png')
    im.save(filename_png)
    os.remove(filename_jpg)

    filename = os.path.join(save_dir, 'depth_0')
    np.save(filename, np.float32(depth_image.reshape([IM_HEIGHT,IM_WIDTH])), False, True)

class GQCNNExecution(object):

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
        self.sim.set_object_pose(stable_pose[:3].flatten())

    def collect_image(self, camera_pose, im_height, im_width):
        self.sim.set_camera_pose(camera_pose)
        self.sim.set_camera_resolution(im_height, im_width)
        rgb_image, depth_image = self.sim.camera_images()
        
        depth_image = np.float32(depth_image.reshape([im_height,im_width,1]))
        rgb_image = np.uint8(rgb_image.reshape([im_height,im_width,3]))
        
        return rgb_image, depth_image

    def check_collision(self, gripper_pose=None):
        collision = self.sim.set_gripper_pose(gripper_pose)
        return collision

    def stop(self):
        self.sim.stop()
        return

if __name__ == '__main__':
    # gqcnn_exec = GQCNNExecution(MESH_PATH)
    # gqcnn_exec.load_new_object()
    # db = SDatabase("/home/silvia/dex-net/data/datasets/silvia_final.hdf5", "main")
    # pose_id = 0
    # object_name = "example1"
    # stable_pose = db.get_stable_pose(object_name, pose_id)
    # gqcnn_exec.drop_object(stable_pose)
    # camera_pose = np.eye(4,4)
    # camera_pose[2, 3] = 0.5
    # camera_pose[1, 1] = -1
    # camera_pose[2, 2] = -1
    # rgb_image, depth_image = gqcnn_exec.collect_image(camera_pose, IM_HEIGHT, IM_WIDTH)
    # save_dir = '/home/silvia/dex-net/planning/'
    # save_images(rgb_image, depth_image, 'color_0', save_dir)
    # depth_im_filename = '/home/silvia/dex-net/planning/depth_0.npy'
    # segmask_filename = None
    # camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir.intr"
    # model_dir = None
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    # grasp = policy.policy_vrep('GQCNN-4.0-PJ', depth_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename)
    # gripper_pose = grasp.pose()
    # gripper_matrix = np.eye(4,4)
    # gripper_matrix[:3,:3] = gripper_pose.rotation
    # gripper_matrix[:3, 3] = gripper_pose.translation

    # gripper_matrix = np.matmul(camera_pose, gripper_matrix)
    # gqcnn_exec.sim.set_gripper_pose(gripper_matrix)

    # gqcnn_exec.stop()
    gqcnn_exec = GQCNNExecution(MESH_PATH)
    gqcnn_exec.load_new_object()
    db = SDatabase("/home/silvia/dex-net/data/datasets/silvia_final.hdf5", "main")
    pose_id = 0
    object_name = "example1"
    stable_pose = db.get_stable_pose(object_name, pose_id)
    gqcnn_exec.drop_object(stable_pose)
    camera_pose = np.eye(4,4)
    camera_pose[2, 3] = 0.5
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1
    rgb_image, depth_image = gqcnn_exec.collect_image(camera_pose, IM_HEIGHT, IM_WIDTH)
    save_dir = '/home/silvia/dex-net/planning/'
    save_images(rgb_image, np.flip(depth_image,1), 'color_0', save_dir)
    depth_im_filename = '/home/silvia/dex-net/planning/depth_0.npy'
    segmask_filename = None
    camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir.intr"
    model_dir = None
    config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    grasp = policy.policy_vrep('GQCNN-4.0-PJ', depth_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename)
    gripper_pose = grasp.pose()
    gripper_matrix = np.eye(4,4)
    gripper_matrix[:3,:3] = gripper_pose.rotation
    gripper_matrix[:3, 3] = gripper_pose.translation

    gripper_matrix = np.matmul(camera_pose, gripper_matrix)
    gqcnn_exec.sim.set_grasp_target(gripper_matrix)
    
    print("sending signal")
    grasp_res = gqcnn_exec.sim.run_threaded_candidate()

    gqcnn_exec.stop()