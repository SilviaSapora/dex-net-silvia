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

if __name__ == '__main__':

    # get mesh and export it to DEFAULT_MESH_PATH
    # db = SDatabase("/home/silvia/dex-net/data/datasets/silvia_procedural_shapes.hdf5", "main")
    # db = SDatabase("/home/silvia/dex-net/data/datasets/dexnet_2_database.hdf5", "3dnet")
    db = SDatabase("/home/silvia/dex-net/data/datasets/dexnet_2_database.hdf5", "kit")

    mesh_dir = "/home/silvia/Downloads/meshes/dexnet_1.0_raw_meshes/PrincetonShapeBenchmark"
    # mesh_dir = "/home/silvia/Downloads/meshes/dexnet_1.0_raw_meshes/amazon_picking_challenge"
    # mesh_dir = "/home/silvia/Desktop/procedurally_generated"
    meshes_from_dir = listdir(mesh_dir)
    meshes_from_dir.sort()

    # Load the mesh from file here, so we can generate grasp candidates
    # and access object-specifsc properties like inertia.
    #mesh = load_mesh(mesh_path)

    res_file = open("/home/silvia/dex-net/policy_res/policy_res_color_unknown.txt","a") 

    config_filename1 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_10.yaml"
    config_filename2 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_20.yaml"
    config_filename3 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_50.yaml"
    config_filename4 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_80.yaml"
    config_filename5 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_2.0_100.yaml"
    config_filename6 = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    config_filename7 = "/home/silvia/dex-net/cfg/examples/dex-net_2.0_color.yaml"
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/examples/replication/dex-net_4.0_pj.yaml"
    # config_filename = "/home/silvia/dex-net/deps/gqcnn/cfg/tools/run_policy_vrep.yaml"
    # model_name = 'gqcnn_spfluv2'
    # model_name = 'GQCNN-4.0-PJ'
    # model_name = 'GQCNN-2.0'
    model_names = ['GQCNN-2.0-10', 'GQCNN-2.0-20', 'GQCNN-2.0-50', 'GQCNN-2.0-80', 'GQCNN-2.0', 'dex-net_4.0_pj', 'GQCNN-2.0-color']
    config_filenames = [config_filename1, config_filename2, config_filename3, config_filename4,config_filename5, config_filename6, config_filename7]
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

    for model_number in range(6,7):
        model_name = model_names[model_number]
        config_filename = config_filenames[model_number]
        for i in range(1,100):
            obj_num = i

            # =========================== OBJ MESH FROM DATABASE ==========================
            keys = db.get_object_keys()
            object_name = keys[obj_num]
            print("obj " + str(obj_num) + ", obj name: " + object_name)
            mesh = db.get_object_mesh(object_name)
            mesh.trimesh.export(DEFAULT_MESH_PATH)
            pose_id = 0
            print("pose id: " + str(pose_id))
            stable_pose = db.get_stable_pose_transform(object_name, pose_id)

            # =========================== OBJ MESH FROM DATABASE ==========================

            # =========================== OBJ MESH FROM DIRECTORY ==========================
            # gqcnn_exec.set_mesh_path(os.path.join(mesh_dir,meshes_from_dir[i]))
            # object_name = meshes_from_dir[i]
            # pose_id = 0
            # print("obj name: " + object_name)
            # print("pose id: " + str(pose_id))
            # stable_pose = np.eye(4,4)
            # stable_pose[2,3] = 0.3
            # =========================== OBJ MESH FROM DIRECTORY ==========================

            # start simulation

            try:
                action = policy.policy_vrep(model_name, depth_im_filename, color_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename)
                # grasp = policy.policy_vrep(model_name, depth_im_filename, segmask_filename, camera_intr_filename, "/home/silvia/dex-net/deps/gqcnn/models", config_filename)
            # if grasp == None:
            except:
                print("obj " + str(obj_num) + " no grasps found")
                res_file.write("{:50s} {:4d} {:25s} No grasp found \n".format(model_name, obj_num, object_name))
                time.sleep(10)
                continue

            grasp = action.grasp
            image = action.image
            q_value = action.q_value
            # q_value = 0
            

            if success == "0":
                success = True
            else:
                success = False
            print("success: " + str(success))
            print("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name, obj_num, object_name, q_value, success))
            res_file.write("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name, obj_num, object_name, q_value, success))
        
            time.sleep(10)
            if (i % 11 == 0):
                time.sleep(60)
        time.sleep(300)

    res_file.close()
    