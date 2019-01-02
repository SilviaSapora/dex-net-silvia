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
from perception import CameraIntrinsics

from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, GraspableObject3D, UniformGraspSampler, AntipodalGraspSampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper, PointGraspMetrics3D

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from constantsTest import *
from dexnet.visualization import DexNetVisualizer3D as vis


CONFIG = YamlConfig(TEST_CONFIG_NAME)

class GraspTest():
    def antipodal_grasp_sampler(self):
    	of = ObjFile(OBJ_FILENAME)
    	sf = SdfFile(SDF_FILENAME)
    	mesh = of.read()
    	sdf = sf.read()
        obj = GraspableObject3D(sdf, mesh)

        gripper = RobotGripper.load(GRIPPER_NAME)

        ags = AntipodalGraspSampler(gripper, CONFIG)

        #print(obj.sdf.surface_points(False))
        #points, _ = obj.sdf.surface_points(False)
        #nparraypoints = np.swapaxes(np.array(points), 0, 1)
        #print(nparraypoints.shape[0])
        
        
        grasps = ags.generate_grasps(obj, target_num_grasps=10)
  
        quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['robust_ferrari_canny'])
        quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, quality_config)
            
        i = 0
        vis.figure()
        #vis.points(PointCloud(nparraypoints), scale=0.001, color=np.array([0.5,0.5,0.5]))
        #vis.plot3d(nparraypoints)
        #vis.mesh(obj.mesh.trimesh, style='surface')
 
        #print(obj)
        #print(obj.mesh)
        #print(obj.sdf.center)
        #print('///////////// SDF SURFACE POINTS TRUE')
        #print(obj.sdf.surface_points(True))
        #print('///////////// SDF SURFACE POINTS FALSE')
        #print(obj.sdf.surface_points(False))
        #print(obj.mesh.trimesh)
        #print('---------------MESH VERTICES--------------')
        #print(obj.mesh.trimesh.vertices)
        #print('---------------END MESH VERTICES--------------')
        low = np.min(CONFIG['metrics'])
        high = np.max(CONFIG['metrics'])
        if low == high:
            q_to_c = lambda quality: CONFIG['quality_scale']
        else:
            q_to_c = lambda quality: CONFIG['quality_scale'] * (quality - low) / (high - low)


        for grasp in grasps:
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                fn_fc = quality_fn(grasp).quality
                true_fc = PointGraspMetrics3D.force_closure(c1, c2, quality_config.friction_coef)
                print(fn_fc)
                #print('fn_fc: ', fn_fc, 'true_fc: ', true_fc)
                #print(grasp.frame)
                #print(grasp)
                #print(grasp.T_grasp_obj)
                #print(grasp.center[0], grasp.center[1], grasp.center[2])
    	    
                #print 'Grasp %d %s=%.5f' %(grasp.id, metric_name, metric)
    	    #T_obj_world = RigidTransform(from_frame='obj',to_frame='world')
    	    #color = quality_fn(grasp).quality
    	    #T_obj_gripper = grasp.gripper_pose(gripper)
            #print('metric: ', CONFIG['metrics']['force_closure'])
            color = plt.get_cmap('hsv')(q_to_c(fn_fc))[:-1]
                #print(grasp.T_grasp_obj)
            print(color)
    	    vis.grasp(grasp, grasp_axis_color=color,endpoint_color=color)
    	    i += 1
        vis.show(False)


if __name__ == '__main__':
    grasp = GraspTest()
    grasp.antipodal_grasp_sampler() 
