import os
import sys
import glob
import h5py
import numpy as np
import trimesh
import time
import math
from os import listdir
from os.path import isfile, join
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

from autolab_core import YamlConfig
import gqcnn
import gqcnn.policy_vrep as policy

DEFAULT_MESH_PATH = "/home/silvia/dex-net/mesh.obj"
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

    def set_mesh_path(self, mesh_path):
        self.mesh_path = mesh_path

    def load_new_object(self, scale=1.0):
        mesh = load_mesh(self.mesh_path)

        mass = mesh.mass_properties['mass'] * 10
        com = mesh.mass_properties['center_mass']
        inertia = mesh.mass_properties['inertia'] * 5

        self.sim.load_object(self.mesh_path, com, mass, inertia.flatten(), scale=scale)

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

    # get mesh and export it to DEFAULT_MESH_PATH
    # db = SDatabase("/home/silvia/dex-net/data/datasets/silvia_procedural_shapes.hdf5", "main")
    # db_train = SDatabase("/home/silvia/dex-net/data/datasets/dexnet_2_database.hdf5", "3dnet")
    # db_test = SDatabase("/home/silvia/dex-net/data/datasets/dexnet_2_database.hdf5", "kit")

    # mesh_dir = "/home/silvia/Downloads/meshes/dexnet_1.0_raw_meshes/PrincetonShapeBenchmark"
    # mesh_dir = "/home/silvia/Downloads/meshes/dexnet_1.0_raw_meshes/amazon_picking_challenge"
    mesh_dir_procedural = "/home/silvia/Desktop/procedurally_generated"
    mesh_dir_priceton = '/home/silvia/Downloads/meshes/dexnet_1.0_raw_meshes/PrincetonShapeBenchmark'

    # data_source = [(True, "3dnet"), (True, "kit"), (False, mesh_dir_procedural), (False, mesh_dir_priceton)]
    # data_source = [(True, "3dnet")]
    # data_source = [(True, "kit")]
    # data_source = [(False, mesh_dir_priceton)]
    data_source = [(True, "3dnet")]

    # Load the mesh from file here, so we can generate grasp candidates
    # and access object-specifsc properties like inertia.
    #mesh = load_mesh(mesh_path)

    res_file = open("/home/silvia/dex-net/policy_res/policy_res_190613.txt","a") 
    # res_file = open("/home/silvia/dex-net/policy_res/policy_res_procedural.txt","a") 

    config_filename1 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_10.yaml"
    config_filename2 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_20.yaml"
    config_filename3 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_50.yaml"
    config_filename4 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_80.yaml"
    config_filename5 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_100.yaml"
    config_filename6 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    config_filename7 = "/home/silvia/dex-net/cfg/examples/dex-net_2.0_color.yaml"
    config_filename8 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_50_procedural.yaml"
    config_filename9 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0-5333-32.yaml"
    config_filename10 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0d-ferrari.yaml"
    config_filename11 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0d-force_closure.yaml"
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/tools/run_policy_vrep.yaml"
    # model_name = 'gqcnn_spfluv2'
    # model_name = 'GQCNN-4.0-PJ'
    # model_name = 'GQCNN-2.0'
    model_names =  ['GQCNN-2.0-10',               # 0
                    'GQCNN-2.0-20',               # 1
                    'GQCNN-2.0-50',               # 2
                    'GQCNN-2.0-80',               # 3
                    'GQCNN-2.0',                  # 4
                    'GQCNN-2.0d-96-fixed',        # 5
                    'dex-net_4.0_pj',             # 6
                    'GQCNN-2.0-color',            # 7
                    'GQCNN-50-procedural',        # 8
                    'GQCNN-2.0-5333-32',          # 9
                    'GQCNN-2.0d-ferrari',         # 10
                    'GQCNN-2.0d-force_closure']   # 11
    config_filenames = [config_filename1, 
                        config_filename2, 
                        config_filename3, 
                        config_filename4,
                        config_filename5,
                        config_filename6, 
                        config_filename6, 
                        config_filename7,
                        config_filename8,
                        config_filename9,
                        config_filename10,
                        config_filename11]
    is_color         = [False, 
                        False, 
                        False, 
                        False,
                        False,
                        False, 
                        False, 
                        True,
                        False,
                        False,
                        False,
                        False]
    is_random         = [True, 
                        False, 
                        False, 
                        False,
                        False,
                        False, 
                        False, 
                        False,
                        False,
                        False,
                        False,
                        False]
    camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir.intr"
    
    save_dir = '/home/silvia/dex-net/planning/'
    depth_im_filename = '/home/silvia/dex-net/planning/depth_0.npy'
    color_im_filename = '/home/silvia/dex-net/planning/color_0.png'
    segmask_filename = None
    model_dir = None

    camera_pose = np.eye(4,4)
    camera_pose[2, 3] = 0.7
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1


    # gqcnn_exec = GQCNNExecution(DEFAULT_MESH_PATH)
    for model_number in [4]:
        for is_dataset, source in data_source:
            if not is_dataset:
                meshes_from_dir = listdir(source)
                meshes_from_dir.sort()
            model_name = model_names[model_number]
            config_filename = config_filenames[model_number]
            color_policy = is_color[model_number]
            random_policy = is_random[model_number]
            for i in range(1,100):
                gqcnn_exec = GQCNNExecution(DEFAULT_MESH_PATH)
                obj_num = i

                if is_dataset:
                    print("Dataset:" + str(source))
                    db = SDatabase("/home/silvia/dex-net/data/datasets/dexnet_2_database.hdf5", source)
                    # =========================== OBJ MESH FROM DATABASE ==========================
                    scale = 1.0
                    keys = db.get_object_keys()
                    object_name = keys[obj_num]
                    print("obj " + str(obj_num) + ", obj name: " + object_name)
                    mesh = db.get_object_mesh(object_name)
                    mesh.trimesh.export(DEFAULT_MESH_PATH)
                    pose_id = 0
                    print("pose id: " + str(pose_id))
                    stable_pose = db.get_stable_pose_transform(object_name, pose_id)

                    T_obj_stp = stable_pose.T_obj_world
                    T_obj_stp = mesh.get_T_surface_obj(T_obj_stp)
                    stable_pose = np.eye(4,4)
                    stable_pose[:3,:3] = T_obj_stp.rotation
                    stable_pose[:3, 3] = T_obj_stp.translation
                    # =========================== OBJ MESH FROM DATABASE ==========================

                else:
                    if source == mesh_dir_procedural:
                        scale = 0.01 # princeton
                    elif source == mesh_dir_priceton:
                        scale = 0.5 # procedural
                    mesh_dir = source
                    # =========================== OBJ MESH FROM DIRECTORY ==========================
                    gqcnn_exec.set_mesh_path(os.path.join(mesh_dir,meshes_from_dir[i]))
                    object_name = meshes_from_dir[i] # procedural
                    # object_name = meshes_from_dir[i*10] # princeton 1 (mistake: names not actual objs)
                    # object_name = meshes_from_dir[i*15+1] # princeton 2 (mistake: names not actual objs)
                    pose_id = 0
                    print("obj name: " + object_name)
                    print("pose id: " + str(pose_id))
                    stable_pose = np.eye(4,4)
                    stable_pose[2,3] = 0.3
                    # =========================== OBJ MESH FROM DIRECTORY ==========================

                # start simulation
                gqcnn_exec.load_new_object(scale)
                gqcnn_exec.drop_object(stable_pose)

                # wait to take picture until object stabilizes
                object_vel = gqcnn_exec.sim.get_object_velocity()
                while abs(object_vel[0]) > 0.0004 or abs(object_vel[1]) > 0.0004 or abs(object_vel[2]) > 0.0004:
                    object_vel = gqcnn_exec.sim.get_object_velocity()
                    # print(object_vel)

                rgb_image, depth_image = gqcnn_exec.collect_image(camera_pose, IM_HEIGHT, IM_WIDTH)
                save_images(np.flip(rgb_image,1), np.flip(depth_image,1), 'color_0', save_dir)
                

                if random_policy:
                    grasp = policy.policy_vrep(model_name, depth_im_filename, color_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename, random=True)
                    if grasp == None:
                        print("obj " + str(obj_num) + " no grasps found")
                        res_file.write("{:50s} {:4d} {:25s} No grasp found \n".format(model_name, obj_num, object_name))
                        gqcnn_exec.stop()
                        time.sleep(10)
                        continue
                    q_value = 0
                else:
                    try:
                        action = policy.policy_vrep(model_name, depth_im_filename, color_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename, random=False)
                        grasp = action.grasp
                        image = action.image
                        q_value = action.q_value
                    except:
                        print("obj " + str(obj_num) + " no grasps found")
                        res_file.write("{:50s} {:4d} {:25s} No grasp found \n".format(model_name, obj_num, object_name))
                        gqcnn_exec.stop()
                        time.sleep(10)
                        continue


                gripper_pose = grasp.pose()
                gripper_matrix = np.eye(4,4)
                gripper_matrix[:3,:3] = gripper_pose.rotation
                gripper_matrix[:3, 3] = gripper_pose.translation

                gripper_matrix = np.matmul(camera_pose, gripper_matrix)
                if color_policy:
                    gripper_matrix[2,3] = 0.015
                else:
                    if gripper_matrix[2,3] < 0.015:
                        gripper_matrix[2,3] = 0.015
                
                gqcnn_exec.sim.set_grasp_target(gripper_matrix)

                success = gqcnn_exec.sim.run_threaded_candidate()
                gqcnn_exec.stop()
                
                if success == "0":
                    success = True
                else:
                    success = False
                
                print("success: " + str(success))
                print("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name, obj_num, object_name, q_value, success))
                res_file.write("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name, obj_num, object_name, q_value, success))
            
                time.sleep(7)
                # if (i % 10 == 0 and i != 0):
                    # print("sleep")
                    # time.sleep(60)
            # time.sleep(240)

    res_file.close()
    