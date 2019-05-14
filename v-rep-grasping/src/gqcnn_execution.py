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
from random import randint

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
            'scene_path': '/home/silvia/dex-net/v-rep-grasping/scenes/sceneIK.ttt',
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

    # get mesh and export it to MESH_PATH
    # db = SDatabase("/home/silvia/dex-net/data/datasets/silvia_procedural_shapes.hdf5", "main")
    db = SDatabase("/home/silvia/dex-net/data/datasets/dexnet_2_database.hdf5", "3dnet")
    # MESH_PATH = "/home/silvia/dex-net/.dexnet/mini_dexnet/bar_clamp.obj"
    # MESH_PATH = "/home/silvia/dex-net/.dexnet/main/example2.obj"
    res_file = open("/home/silvia/dex-net/policy_res/policy_res2.txt","a") 

    config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0.yaml"
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/tools/run_policy_vrep.yaml"
    model_name = 'gqcnn_spfluv2'
    # model_name = 'GQCNN-4.0-PJ'
    # model_name = 'GQCNN-2.0'
    camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir.intr"
    
    save_dir = '/home/silvia/dex-net/planning/'
    depth_im_filename = '/home/silvia/dex-net/planning/depth_0.npy'
    segmask_filename = None
    model_dir = None

    camera_pose = np.eye(4,4)
    camera_pose[2, 3] = 0.7
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1

    for i in range(21,22):
        gqcnn_exec = GQCNNExecution(MESH_PATH)
        keys = db.get_object_keys()
        obj_num = i
        object_name = keys[obj_num]
        print(object_name)
        mesh = db.get_object_mesh(object_name)
        mesh.trimesh.export(MESH_PATH)
        pose_id = 1
        print("pose id: " + str(pose_id))
        stable_pose = db.get_stable_pose(object_name, pose_id)
        # stable_pose = np.eye(4,4)
        # stable_pose[2,3] = 0.2

        # start simulation
        gqcnn_exec.load_new_object()
        gqcnn_exec.drop_object(stable_pose)

        # wait to take picture until object stabilizes
        object_vel = gqcnn_exec.sim.get_object_velocity()
        while abs(object_vel[0]) > 0.0002 or abs(object_vel[1]) > 0.0002 or abs(object_vel[2]) > 0.0002:
            object_vel = gqcnn_exec.sim.get_object_velocity()

        rgb_image, depth_image = gqcnn_exec.collect_image(camera_pose, IM_HEIGHT, IM_WIDTH)
        save_images(rgb_image, np.flip(depth_image,1), 'color_0', save_dir)
        
        try:
            action = policy.policy_vrep(model_name, depth_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename)
        except:
            print("obj " + str(obj_num) + " no grasps found")
            res_file.write("{:50s} {:4d} {:25s} No grasp found \n".format(model_name, obj_num, object_name, q_value, success))
            gqcnn_exec.stop()
            time.sleep(60)
            continue

        grasp = action.grasp
        image = action.image
        q_value = action.q_value
        gripper_pose = grasp.pose()
        gripper_matrix = np.eye(4,4)
        gripper_matrix[:3,:3] = gripper_pose.rotation
        gripper_matrix[:3, 3] = gripper_pose.translation

        gripper_matrix = np.matmul(camera_pose, gripper_matrix)
        if gripper_matrix[2,3] < 0.015:
            gripper_matrix[2,3] = 0.015
        
        gqcnn_exec.sim.set_grasp_target(gripper_matrix)

        success = gqcnn_exec.sim.run_threaded_candidate()
        if success == "0":
            success = True
        else:
            success = False
        print("success: " + str(not success))
        print("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name, obj_num, object_name, q_value, success))
        res_file.write("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name, obj_num, object_name, q_value, success))
    
        gqcnn_exec.stop()
        time.sleep(60)

    res_file.close()
    