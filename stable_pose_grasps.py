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

from autolab_core import RigidTransform, YamlConfig, BagOfPoints, PointCloud

from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, GraspableObject3D, UniformGraspSampler, AntipodalGraspSampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper, PointGraspMetrics3D

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from dexnet.database import MeshProcessor, RescalingType
from constantsTest import *
from dexnet.visualization import DexNetVisualizer3D as vis
from visualization import Visualizer3D


CONFIG = YamlConfig(TEST_CONFIG_NAME)

class GraspTest():
    def antipodal_grasp_sampler(self):
        mass = 1.0
        CONFIG['obj_rescaling_type'] = RescalingType.RELATIVE
        mesh_processor = MeshProcessor(OBJ_FILENAME, CONFIG['cache_dir'])
        mesh_processor.generate_graspable(CONFIG)
        mesh = mesh_processor.mesh
        sdf = mesh_processor.sdf
        stable_poses = mesh_processor.stable_poses
        obj = GraspableObject3D(sdf, mesh)
        print(len(stable_poses))
        #for stable_pose in stable_poses:
        #    print(stable_pose.p)
        #    print(stable_pose.r)
        #    print(stable_pose.x0)
        #    print(stable_pose.face)
        #    print(stable_pose.id)
        stable_pose = stable_poses[0]
        #print(stable_pose.p)
        print(stable_pose.r)
        #print(stable_pose.x0)
        print(stable_pose.T_obj_world)
        gripper = RobotGripper.load(GRIPPER_NAME)

        ags = AntipodalGraspSampler(gripper, CONFIG)
        stable_pose.id = 0
        grasps = ags.generate_grasps(obj,target_num_grasps=10, max_iter=5)
  
        quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['robust_ferrari_canny'])

        vis.figure()
        metrics = []
        
        #vis.mesh(obj.mesh.trimesh, style='surface')
        for grasp in grasps:
            angle = grasp.grasp_angles_from_stp_z(stable_pose)
            print(angle)
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                print("Grasp:")
                print(c1.point)
                print(c2.point)
                true_fc = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config)
                true_fc = true_fc
                metrics.append(true_fc)

        low = np.min(metrics)
        high = np.max(metrics)
        if low == high:
            q_to_c = lambda quality: CONFIG['quality_scale']
        else:
            q_to_c = lambda quality: CONFIG['quality_scale'] * (quality - low) / (high - low)
        
        print(len(metrics))
        T_table_world=RigidTransform(from_frame='table', to_frame='world')
        T_obj_world = Visualizer3D.mesh_stable_pose(obj.mesh.trimesh, stable_pose.T_obj_world, 
                                                    T_table_world=T_table_world, color=(0.5,0.5,0.5), 
                                                    style='surface', plot_table=True, dim=0.15)

        for grasp, metric in zip(grasps, metrics):
            grasp = grasp.perpendicular_table(stable_pose)
            color = plt.get_cmap('hsv')(q_to_c(metric))[:-1]
            vis.grasp(grasp, T_obj_world=T_obj_world, grasp_axis_color=color,endpoint_color=color)

        vis.show(False)


if __name__ == '__main__':
    grasp = GraspTest()
    grasp.antipodal_grasp_sampler() 
