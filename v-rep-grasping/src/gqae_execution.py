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
import torch.utils.data

sys.path.insert(0, '/home/silvia/dex-net/v-rep-grasping/')

import lib
import lib.utils
from lib.config import config_mesh_dir, config_output_collected_dir
import vrep
vrep.simxFinish(-1)
import simulator as SI

sys.path.insert(0, '/home/silvia/dex-net/')

from store_grasps_old_db import SDatabase

from autolab_core import YamlConfig, Point
from perception import CameraIntrinsics, DepthImage
import gqcnn
import gqcnn.policy_vrep as policy
from gqcnn.grasping import Grasp2D

sys.path.insert(0, '/home/silvia/ggcnn/')
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from models import get_network

DEFAULT_MESH_PATH = "/home/silvia/dex-net/mesh.obj"
IM_HEIGHT = 400
IM_WIDTH = 400

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

class GQAEExecution(object):

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

    # data_source = [(True, "3dnet"), (True, "kit"), (False, mesh_dir_procedural), (False, mesh_dir_procedural)]
    data_source = [(True, "3dnet")]
    # data_source = [(False, mesh_dir_priceton)]
    # data_source = [(False, mesh_dir_procedural)]

    # Load the mesh from file here, so we can generate grasp candidates
    # and access object-specifsc properties like inertia.
    #mesh = load_mesh(mesh_path)

    res_file = open("/home/silvia/dex-net/policy_res/gqae_res_fixed.txt","a")

    model_names =  [
                    '/home/silvia/fc_ggcnn/models/190614_1840_fc_6_angles_fc_5/epoch_10_iou_0.00',
                    # '/home/silvia/fc_ggcnn/models/190614_1305_fc_6_ignore_angled_finetune_BCE/epoch_21_iou_0.00',
                    # '/home/silvia/fc_ggcnn/models/190614_1840_fc_6_angles_fc_5/epoch_20_iou_0.00'
                    ]
    model_name_s = [
                    'fc_6_angles_fc_5/epoch_10', 
                    # 'fc_6_finetuned/epoch_21', 
                    # 'fc_6_angles_fc_5/epoch_20'
                    ]
    # camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir.intr"
    camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir_gqae.intr"
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    save_dir = '/home/silvia/dex-net/planning/'
    depth_im_filename = '/home/silvia/dex-net/planning/depth_0.npy'
    color_im_filename = '/home/silvia/dex-net/planning/color_0.png'
    segmask_filename = None
    model_dir = None

    camera_pose = np.eye(4,4)
    camera_pose[2, 3] = 0.7
    camera_pose[1, 1] = -1
    camera_pose[2, 2] = -1

    device = torch.device("cpu")

    for model_number in range(1):
        model_name = model_names[model_number]
        # ggcnn = get_network('ggcnn3')
        model = torch.load(model_name, map_location='cpu')

        for is_dataset, source in data_source:
            if not is_dataset:
                meshes_from_dir = listdir(source)
                meshes_from_dir.sort()
            
            for i in range(1,10):
                gqae_exec = GQAEExecution(DEFAULT_MESH_PATH)
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
                    gqae_exec.set_mesh_path(os.path.join(mesh_dir,meshes_from_dir[i]))
                    object_name = meshes_from_dir[i] # procedural
                    # object_name = meshes_from_dir[i*10] # princeton 1
                    # object_name = meshes_from_dir[i*15+1] # princeton 2
                    pose_id = 0
                    print("obj name: " + object_name)
                    print("pose id: " + str(pose_id))
                    stable_pose = np.eye(4,4)
                    stable_pose[2,3] = 0.3
                    # =========================== OBJ MESH FROM DIRECTORY ==========================

                # start simulation
                gqae_exec.load_new_object(scale)
                gqae_exec.drop_object(stable_pose)

                # wait to take picture until object stabilizes
                object_vel = gqae_exec.sim.get_object_velocity()
                while abs(object_vel[0]) > 0.0004 or abs(object_vel[1]) > 0.0004 or abs(object_vel[2]) > 0.0004:
                    object_vel = gqae_exec.sim.get_object_velocity()

                rgb_image, depth_image = gqae_exec.collect_image(camera_pose, IM_HEIGHT, IM_WIDTH)
                # save_images(np.flip(rgb_image,1), np.flip(depth_image,1), 'color_0', save_dir)
                # depth_image = np.flip(depth_image,1)
                depth_im = DepthImage(np.flip(depth_image,1)).resize((200, 200))
                # depth_im = DepthImage(depth_image).resize((200, 200))
                depth_tensor = torch.from_numpy(np.expand_dims(depth_im.raw_data.reshape((200,200)), 0).astype(np.float32)).reshape((1,1,200,200))

                xc = depth_tensor.to(device)
                pos_pred, cos_pred, sin_pred = model(xc)

                # x_n = xc.detach().numpy().reshape(200,200)
                pos_pred_n = pos_pred.detach().numpy().reshape(200,200)
                cos_pred_n = cos_pred.detach().numpy().reshape(200,200)
                sin_pred_n = sin_pred.detach().numpy().reshape(200,200)

                # plt.figure(figsize=(14, 4))
                # plt.subplot(131)
                # plt.title("QUALITY")
                # plt.imshow(pos_pred_n)
                # plt.colorbar()
                # plt.subplot(132)
                # plt.title("COS")
                # plt.imshow(cos_pred_n)
                # plt.colorbar()
                # plt.subplot(133)
                # plt.title("SIN")
                # plt.imshow(sin_pred_n)
                # plt.colorbar()
                # plt.show()

                # print("COORDS")
                idx = np.argmax(pos_pred_n)
                # print(idx)
                x = idx / 200
                y = idx % 200
                # print(x)
                # print(y)
                # print("MAX VALUE")
                # print(np.max(pos_pred_n))
                q_value = pos_pred_n[x,y]
                # print(q_value)
                grasp_angle = 0.5 * math.atan2(sin_pred_n[x,y], cos_pred_n[x,y])
                grasp_depth = depth_im.raw_data[x,y] + 0.03

                coords_y = ((y-100) * 2) + 100
                coords_x = ((x-100) * 2) + 100
                grasp = Grasp2D(Point(np.array([coords_y,coords_x])), grasp_angle, grasp_depth,
                                     width = 50,
                                     camera_intr=camera_intr)
                
                # grasp_angle_im = np.zeros((200,200))
                # for xx in range(200):
                    # for yy in range(200):
                        # grasp_angle_im[xx,yy] = 0.5 * math.atan2(sin_pred_n[xx,yy], cos_pred_n[xx,yy])
                
                # axis = np.array([np.cos(grasp_angle_im[x,y]), np.sin(grasp_angle_im[x,y])])
                # center = [y,x]
                # g1p = center - (axis * 10) # start location of grasp jaw 1
                # g2p = center + (axis * 10) # start location of grasp jaw 2
                # plt.figure(figsize=(14, 4))
                # plt.subplot(131)
                # plt.title("DEPTH")
                # plt.imshow(x_n)
                # plt.colorbar()
                # plt.subplot(132)
                # plt.title("QUALITY")
                # plt.imshow(pos_pred_n)
                # plt.colorbar()
                # plt.subplot(133)
                # plt.title("ANGLE")
                # plt.imshow(grasp_angle_im)
                # plt.colorbar()
                # plt.plot([g1p[0], g2p[0]], [g1p[1], g2p[1]], color='firebrick', linewidth=5, linestyle='--')
                # plt.show()

                gripper_pose = grasp.pose()
                gripper_matrix = np.eye(4,4)
                gripper_matrix[:3,:3] = gripper_pose.rotation
                gripper_matrix[:3, 3] = gripper_pose.translation

                gripper_matrix = np.matmul(camera_pose, gripper_matrix)
                if gripper_matrix[2,3] < 0.015:
                    gripper_matrix[2,3] = 0.01
                
                gqae_exec.sim.set_grasp_target(gripper_matrix)

                success = gqae_exec.sim.run_threaded_candidate()
                gqae_exec.stop()
                
                if success == "0":
                    success = True
                else:
                    success = False
                
                print("success: " + str(success))
                print("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name_s[model_number], obj_num, object_name, q_value, success))
                res_file.write("{:50s} {:4d} {:25s} {:5f} {:1d}\n".format(model_name_s[model_number], obj_num, object_name, q_value, success))
            
                time.sleep(7)
                # if (i % 10 == 0 and i != 0):
                    # print("sleep")
                    # time.sleep(60)

    res_file.close()
    