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


CONFIG = YamlConfig(TEST_CONFIG_NAME)

class GraspTest():
    def antipodal_grasp_sampler(self):
    	#of = ObjFile(OBJ_FILENAME)
    	#sf = SdfFile(SDF_FILENAME)
    	#mesh = of.read()
    	#sdf = sf.read()
        #obj = GraspableObject3D(sdf, mesh)
        mass = 1.0
        CONFIG['obj_rescaling_type'] = RescalingType.RELATIVE
        mesh_processor = MeshProcessor(OBJ_FILENAME, CONFIG['cache_dir'])
        mesh_processor.generate_graspable(CONFIG)
        mesh = mesh_processor.mesh
        sdf = mesh_processor.sdf
        obj = GraspableObject3D(sdf, mesh)

        gripper = RobotGripper.load(GRIPPER_NAME)

        ags = AntipodalGraspSampler(gripper, CONFIG)
        grasps = ags.sample_grasps(obj, num_grasps=10)
  
        quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['robust_ferrari_canny'])
        quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, quality_config)
        
        i = 0
        vis.figure()
        #vis.points(PointCloud(nparraypoints), scale=0.001, color=np.array([0.5,0.5,0.5]))
        #vis.plot3d(nparraypoints)

        vis.mesh(obj.mesh.trimesh, style='surface')
        for grasp in grasps:
            success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                #fn_fc = quality_fn(grasp).quality
                true_fc = PointGraspMetrics3D.grasp_quality(grasp, obj, quality_config)
                true_fc = true_fc * 100

                #print 'Grasp %d %s=%.5f' %(grasp.id, metric_name, metric)
    	    #color = quality_fn(grasp).quality
            if (true_fc > 1.0):
                true_fc = 1.0
            color = (1.0-true_fc, true_fc, 0.0)
            #color = plt.get_cmap('hsv')(true_fc)[:-1]
                #print(grasp.T_grasp_obj)
            print(color)
    	    vis.grasp(grasp, grasp_axis_color=color,endpoint_color=color)
    	    i += 1

        vis.show(False)


if __name__ == '__main__':
    grasp = GraspTest()
    grasp.antipodal_grasp_sampler() 
