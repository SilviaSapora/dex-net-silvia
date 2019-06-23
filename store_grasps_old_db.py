# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Tests grasping basic functionality
Author: Jeff Mahler
"""
import copy
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import math

from autolab_core import RigidTransform, YamlConfig, BagOfPoints, PointCloud

from dexnet.grasping import (Contact3D, ParallelJawPtGrasp3D, GraspableObject3D, 
                            UniformGraspSampler, AntipodalGraspSampler, 
                            GraspQualityConfigFactory, GraspQualityFunctionFactory, 
                            RobotGripper, PointGraspMetrics3D)

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.database import MeshProcessor, RescalingType
import dexnet.database.database as db
from constantsTest import *
from dexnet.visualization import DexNetVisualizer3D as vis
from visualization import Visualizer3D
from dexnet import DexNet

sys.path.append('../')
from stable_pose_grasps import antipodal_grasp_sampler_for_storing, grasp_quality_calculator


CONFIG = YamlConfig(TEST_CONFIG_NAME)

def get_contact_points(grasp):
    return grasp.endpoints

class SDatabase(object):

    def __init__(self, database_path, dataset_name):
        self.database_path = database_path
        self.dataset_name = dataset_name

    # returns the grasps associated with given object_name and stable_pose_id
    def read_grasps(self, object_name, stable_pose_id, max_grasps=10, visualize=False):
        # load gripper
        gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        # read the most robust grasp
        sorted_grasps, metrics = dexnet_handle.dataset.sorted_grasps(object_name, stable_pose_id=('pose_'+str(stable_pose_id)), metric='force_closure', gripper=gripper.name)
        
        if (len(sorted_grasps)) == 0:
            print('no grasps for this stable pose')
            if visualize:
                stable_pose = dexnet_handle.dataset.stable_pose(object_name, stable_pose_id=('pose_'+str(stable_pose_id)))
                obj = dexnet_handle.dataset.graspable(object_name)
                vis.figure()
                T_table_world = RigidTransform(from_frame='table', to_frame='world')
                T_obj_world = Visualizer3D.mesh_stable_pose(obj.mesh.trimesh, stable_pose.T_obj_world, 
                                                        T_table_world=T_table_world, color=(0.5,0.5,0.5), 
                                                        style='surface', plot_table=True, dim=0.15)
                vis.show(False)
            return None, None
        
        contact_points = map(get_contact_points, sorted_grasps)
        
        # ------------------------ VISUALIZATION CODE ------------------------
        if visualize:
            low = np.min(metrics)
            high = np.max(metrics)
            if low == high:
                q_to_c = lambda quality: CONFIG['quality_scale']
            else:
                q_to_c = lambda quality: CONFIG['quality_scale'] * (quality - low) / (high - low)

            stable_pose = dexnet_handle.dataset.stable_pose(object_name, stable_pose_id=('pose_'+str(stable_pose_id)))
            obj = dexnet_handle.dataset.graspable(object_name)
            vis.figure()

            T_table_world = RigidTransform(from_frame='table', to_frame='world')
            T_obj_world = Visualizer3D.mesh_stable_pose(obj.mesh.trimesh, stable_pose.T_obj_world, 
                                                    T_table_world=T_table_world, color=(0.5,0.5,0.5), 
                                                    style='surface', plot_table=True, dim=0.15)
            for grasp, metric in zip(sorted_grasps, metrics):
                color = plt.get_cmap('hsv')((q_to_c(metric)))[:-1]
                vis.grasp(grasp, T_obj_world=stable_pose.T_obj_world, grasp_axis_color=color,endpoint_color=color)
            vis.show(False)
        # ------------------------ END VISUALIZATION CODE ---------------------------


        # ------------------------ START COLLISION CHECKING ---------------------------
        #stable_pose = dexnet_handle.dataset.stable_pose(object_name, stable_pose_id=('pose_'+str(stable_pose_id)))
        #graspable = dexnet_handle.dataset.graspable(object_name)
        #cc = GraspCollisionChecker(gripper).set_graspable_object(graspable, stable_pose.T_obj_world)        
        stable_pose_matrix = self.get_stable_pose(object_name, stable_pose_id)
        # CLOSE DATABASE
        dexnet_handle.close_database()
        gripper_poses = []

        for grasp in sorted_grasps[:max_grasps]:
            gripper_pose_matrix = np.eye(4,4)
            center_world = np.matmul(stable_pose_matrix, [grasp.center[0], grasp.center[1], grasp.center[2], 1])
            axis_world = np.matmul(stable_pose_matrix, [grasp.axis_[0], grasp.axis_[1], grasp.axis_[2], 1])
            gripper_angle = math.atan2(axis_world[1], axis_world[0])
            gripper_pose_matrix[:3, 3] = center_world[:3]
            gripper_pose_matrix[0,0] = math.cos(gripper_angle)
            gripper_pose_matrix[0,1] = -math.sin(gripper_angle)
            gripper_pose_matrix[1,0] = math.sin(gripper_angle)
            gripper_pose_matrix[1,1] = math.cos(gripper_angle)
            #if visualize:
            #    vis.figure()
            #    vis.gripper_on_object(gripper, grasp, obj, stable_pose=stable_pose.T_obj_world)
            #    vis.show(False)
            gripper_poses.append(gripper_pose_matrix)

        return contact_points, gripper_poses

    # returns the matrix corresponding to the stable pose
    def get_stable_pose(self, object_name, stable_pose_id):
        # load gripper
        gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        stable_pose = dexnet_handle.dataset.stable_pose(object_name, stable_pose_id=('pose_'+str(stable_pose_id)))

        dexnet_handle.close_database()

        pose_matrix = np.eye(4,4)
        pose_matrix[:3,:3] = stable_pose.T_obj_world.rotation
        pose_matrix[:3, 3] = stable_pose.T_obj_world.translation
        return pose_matrix

    # returns the stable pose
    def get_stable_pose_transform(self, object_name, stable_pose_id):
        # load gripper
        gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        stable_pose = dexnet_handle.dataset.stable_pose(object_name, stable_pose_id=('pose_'+str(stable_pose_id)))

        dexnet_handle.close_database()
        
        return stable_pose

    # returns the mesh of the object with the given key
    def get_object_mesh(self, object_name):
        # load gripper
        gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        object_mesh = dexnet_handle.dataset.mesh(object_name)

        return object_mesh

    # generates grasps for mesh given at filepath and saves them in the database under the name object_name
    def database_save(self, object_name, filepath, stable_poses_n, metric, overwrite_object=False, force_overwrite=False):
        # load gripper
        gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        mass = CONFIG['default_mass']

        if object_name in dexnet_handle.dataset.object_keys:
            if not overwrite_object:
                dexnet_handle.close_database()
                print(object_name + " already exists in database, not overwriting")
                return
            graspable = dexnet_handle.dataset[object_name]
            mesh = graspable.mesh
            sdf = graspable.sdf
            stable_poses = dexnet_handle.dataset.stable_poses(object_name)
        else:
            # Create temp dir if cache dir is not provided
            mp_cache = CONFIG['cache_dir']
            del_cache = False
            if mp_cache is None:
                mp_cache = tempfile.mkdtemp()
                del_cache = True
            
            # open mesh preprocessor
            graspable = MeshProcessor(filepath, mp_cache)
            graspable.generate_graspable(CONFIG)
            mesh = graspable.mesh
            sdf = graspable.sdf
            stable_poses = graspable.stable_poses[:stable_poses_n]

            # write graspable to database
            dexnet_handle.dataset.create_graspable(object_name, mesh, sdf, stable_poses, mass=mass)

        if force_overwrite:
            dexnet_handle.dataset.delete_grasps(object_name, gripper=gripper.name)

        # calculate or retrieve grasps
        loaded_grasps = dexnet_handle.dataset.grasps(object_name, gripper=gripper.name)
        if loaded_grasps == []:
            print("No grasps in database, calculating")
            grasps = antipodal_grasp_sampler_for_storing(graspable)
            if (grasps != None):
                print("Saving grasps")
                dexnet_handle.dataset.store_grasps(object_name, grasps, gripper=gripper.name, force_overwrite=force_overwrite)
                loaded_grasps = dexnet_handle.dataset.grasps(object_name, gripper=gripper.name)

        print("Calculating grasp quality")
        metrics = grasp_quality_calculator(mesh, sdf, loaded_grasps, metric)
        print("Grasp quality calculated")
        grasp_metrics = {}
        for g in loaded_grasps:
            grasp_metrics[g.id] = {}
            grasp_metrics[g.id][metric] = metrics[g.id]
        dexnet_handle.dataset.store_grasp_metrics(object_name, grasp_metrics, gripper=gripper.name)
        dexnet_handle.close_database()

    # Deletes all grasps and stable poses for the given object and gripper
    def database_delete_grasps(self, object_name):
        # load gripper
        gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        dexnet_handle.dataset.delete_grasps(object_name, gripper=gripper.name)

        dexnet_handle.close_database()

    # Deletes graspable
    def delete_graspable(self, object_name):
        # open Dex-Net API
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        dexnet_handle.dataset.delete_graspable(object_name)
        
        # delete files related to object
        path_to_obj_file = ".dexnet/" + object_name
        if os.path.exists(path_to_obj_file + "_proc.sdf"):
            print("sdf file removed")
            os.remove(path_to_obj_file + "_proc.sdf")
        if os.path.exists(path_to_obj_file + "_proc.obj"):
            print("obj file removed")
            os.remove(path_to_obj_file + "_proc.obj")

        dexnet_handle.close_database()

    def clear_database(self):
        for i in range(19):
            object_name = "example" + str(i)
            self.delete_graspable(object_name)

    def get_object_keys(self):
        dexnet_handle = DexNet()
        dexnet_handle.open_database(self.database_path)
        dexnet_handle.open_dataset(self.dataset_name)

        obj_keys = dexnet_handle.dataset.object_keys

        dexnet_handle.close_database()
        return obj_keys

def calculate_grasps(db, mesh_dir, name_format, obj_n, stable_poses_n, metric):
    for i in range(0, obj_n):
        object_name = name_format + str(i)
        print('saving for obj: ' + object_name)
        mesh_path = mesh_dir + object_name + '.obj'
        db.database_save(object_name, mesh_path, stable_poses_n, metric, True, False)            

def read_grasps(db, object_name, pose_id):
    print('pose id: ' + str(pose_id) + ' for obj: ' + str(object_name))
    loaded_grasps = dexnet_handle.dataset.grasps(object_name, gripper=gripper.name)


if __name__ == '__main__':
    # db = SDatabase("example.hdf5", "mini_dexnet")
    print("hello")
    db = SDatabase("rescling_test.hdf5", "main")
    # db.clear_database()
    #db.delete_graspable('example0')
    object_name = "procedural_obj_"
    mesh_dir = '/home/silvia/dex-net/generated_shapes/'
    # mesh_path = "/home/silvia/dex-net/.dexnet/mini_dexnet/bar_clamp.obj"
    # db.delete_graspable(object_name)
    # db.database_save(object_name, mesh_path, 1, "robust_ferrari_canny", True, False)            
    print("hello2")
    db.delete_graspable(object_name + str(0))

    calculate_grasps(db, mesh_dir, object_name, obj_n=1, stable_poses_n=1, metric="force_closure")
    # for obj in range(1):
       # for pose_id in range(1):
           # read_grasps(db, object_name + str(obj), pose_id)
