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
                            UniformGraspSampler, AntipodalGraspSampler, GraspQualityConfigFactory, 
                            GraspQualityFunctionFactory, RobotGripper, PointGraspMetrics3D)

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.database import MeshProcessor, RescalingType
from constantsTest import *
from dexnet.visualization import DexNetVisualizer3D as vis
from visualization import Visualizer3D


CONFIG = YamlConfig(TEST_CONFIG_NAME)

def grasp_is_parallel_to_table(grasp_axis, stable_pose_r):
    angle = np.dot(np.matmul(stable_pose_r, grasp_axis), np.array([0,0,1]))

    # not parallel if angle is greater than 36 degrees
    return abs(angle) < 0.2

def antipodal_grasp_sampler(visual=False, debug=False):
    mass = 1.0
    CONFIG['obj_rescaling_type'] = RescalingType.RELATIVE
    mesh_processor = MeshProcessor(OBJ_FILENAME, CONFIG['cache_dir'])
    mesh_processor.generate_graspable(CONFIG)
    mesh = mesh_processor.mesh
    sdf = mesh_processor.sdf
    stable_poses = mesh_processor.stable_poses
    obj = GraspableObject3D(sdf, mesh)
    stable_pose = stable_poses[0]
    if visual:
        vis.figure()

    T_table_world = RigidTransform(from_frame='table', to_frame='world')
    T_obj_world = Visualizer3D.mesh_stable_pose(obj.mesh.trimesh, stable_pose.T_obj_world, 
                                                T_table_world=T_table_world, color=(0.5,0.5,0.5), 
                                                style='surface', plot_table=True, dim=0.15)
    if debug:
        print(len(stable_poses))
    #for stable_pose in stable_poses:
    #    print(stable_pose.p)
    #    print(stable_pose.r)
    #    print(stable_pose.x0)
    #    print(stable_pose.face)
    #    print(stable_pose.id)
    # glass = 22 is standing straight
    if debug:
        print(stable_pose.r)
        print(stable_pose.T_obj_world)
    gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

    ags = AntipodalGraspSampler(gripper, CONFIG)
    stable_pose.id = 0
    grasps = ags.generate_grasps(obj,target_num_grasps=20, max_iter=5, stable_pose=stable_pose.r)

    quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['robust_ferrari_canny'])

    metrics = []
    result = []
    
    #grasps = map(lambda g : g.perpendicular_table(stable_pose), grasps)

    for grasp in grasps:
        c1, c2 = grasp.endpoints
        true_fc = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config)
        metrics.append(true_fc)
        result.append((c1,c2))

        if debug:
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                print("Grasp:")
                print(c1.point)
                print(c2.point)
                print(true_fc)

    low = np.min(metrics)
    high = np.max(metrics)
    if low == high:
        q_to_c = lambda quality: CONFIG['quality_scale']
    else:
        q_to_c = lambda quality: CONFIG['quality_scale'] * (quality - low) / (high - low)

    if visual:
        for grasp, metric in zip(grasps, metrics):
            #grasp2 = grasp.perpendicular_table(stable_pose)
            #c1, c2 = grasp.endpoints
            #axis = ParallelJawPtGrasp3D.axis_from_endpoints(c1, c2)
            #angle = np.dot(np.matmul(stable_pose.r, axis), [1,0,0])
            #angle = math.tan(axis[1]/axis[0])
            #angle = math.degrees(angle)%360
            #print(angle)
            #print(angle/360.0)
            color = plt.get_cmap('hsv')(q_to_c(metric))[:-1]
            vis.grasp(grasp, T_obj_world=T_obj_world, grasp_axis_color=color,endpoint_color=color)

        #axis = np.array([[0,0,0], point])
        #points = [(x[0], x[1], x[2]) for x in axis]
        #Visualizer3D.plot3d(points, color=(0,0,1), tube_radius=0.002)
        vis.show(False)

    pose_matrix = np.eye(4,4)
    pose_matrix[:3,:3] = T_obj_world.rotation
    pose_matrix[:3, 3] = T_obj_world.translation
    return pose_matrix, result

def antipodal_grasp_sampler_for_storing(mesh, sdf, stable_poses):
    mass = 1.0
    CONFIG['obj_rescaling_type'] = RescalingType.RELATIVE
    obj = GraspableObject3D(sdf, mesh)

    gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

    ags = AntipodalGraspSampler(gripper, CONFIG)

    quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['force_closure'])
    quality_function = GraspQualityFunctionFactory.create_quality_function(obj, quality_config)

    max_poses = len(stable_poses)
    grasps = [None] * max_poses
    metrics = [None] * max_poses
    all_grasps = ags.generate_grasps(obj,target_num_grasps=200, max_iter=4)

    for id, stable_pose in enumerate(stable_poses):
            print('sampling for stable pose: ', id)
            if id == max_poses:
                break
            grasps_pose = filter(lambda x: grasp_is_parallel_to_table(x.axis, stable_pose.r), all_grasps)
            #grasps_pose = ags.generate_grasps(obj,target_num_grasps=20, max_iter=5, stable_pose=stable_pose.r)
            grasps[id] = []
            metrics[id] = []
            for grasp in grasps_pose:
                quality = quality_function.quality(grasp)
                #quality = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config)
                grasps[id].append(copy.deepcopy(grasp))
                metrics[id].append(copy.deepcopy(quality.quality))
    return grasps, metrics

def antipodal_grasp_sampler_for_storing(mesh, sdf):
    mass = 1.0
    CONFIG['obj_rescaling_type'] = RescalingType.RELATIVE
    obj = GraspableObject3D(sdf, mesh)

    gripper = RobotGripper.load(GRIPPER_NAME, gripper_dir='/home/silvia/dex-net/data/grippers')

    ags = AntipodalGraspSampler(gripper, CONFIG)

    quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['force_closure'])
    quality_function = GraspQualityFunctionFactory.create_quality_function(obj, quality_config)

    grasps = []
    metrics = []
    all_grasps = ags.generate_grasps(obj,target_num_grasps=200, max_iter=4)

    for grasp in all_grasps:
        quality = quality_function.quality(grasp)
        grasps.append(copy.deepcopy(grasp))
        metrics.append(copy.deepcopy(quality.quality))
    return grasps, metrics


def contacts_from_grasp(grasp):
    return grasp.endpoints

if __name__ == '__main__':
    antipodal_grasp_sampler() 
