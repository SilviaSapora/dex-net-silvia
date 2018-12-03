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

from autolab_core import RigidTransform, YamlConfig
from perception import CameraIntrinsics

from dexnet.grasping import Contact3D, ParallelJawPtGrasp3D, GraspableObject3D, UniformGraspSampler, AntipodalGraspSampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper, PointGraspMetrics3D

from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile
from constants import *
from dexnet.visualization import DexNetVisualizer3D as vis


CONFIG = YamlConfig(TEST_CONFIG_NAME)

def random_force_closure_test_case(antipodal=False):
	    """ Generates a random contact point pair and surface normal, constraining the points to be antipodal if specified and not antipodal otherwise"""
	    scale = 10
	    contacts = scale * (np.random.rand(3,2) - 0.5)
	    
	    mu = 0.0
	    while mu == 0.0:
		mu = np.random.rand()
	    gamma = 0.0
	    while gamma == 0.0:
		gamma = np.random.rand()
	    num_facets = 3 + 100 * int(np.random.rand())

	    if antipodal:
		tangent_cone_scale = mu
		tangent_cone_add = 0
		n0_mult = 1
		n1_mult = 1
	    else:
		n0_mult = 2 * (np.random.randint(0,2) - 0.5)
		n1_mult = 2 * (np.random.randint(0,2) - 0.5)
		tangent_cone_scale = 10
		tangent_cone_add = mu

		if (n0_mult < 0 or n1_mult < 0) and np.random.rand() > 0.5:
		    tangent_cone_scale = mu
		    tangent_cone_add = 0
		    
	    v = contacts[:,1] - contacts[:,0]
	    normals = np.array([-v, v]).T
	    normals = normals / np.tile(np.linalg.norm(normals, axis=0), [3,1])
	    
	    U, _, _ = np.linalg.svd(normals[:,0].reshape(3,1))
	    beta = tangent_cone_scale * np.random.rand() + tangent_cone_add
	    theta = 2 * np.pi * np.random.rand()
	    normals[:,0] = n0_mult * normals[:,0] + beta * np.sin(theta) * U[:,1] + beta * np.cos(theta) * U[:,2]

	    U, _, _ = np.linalg.svd(normals[:,1].reshape(3,1))
	    beta = tangent_cone_scale * np.random.rand() + tangent_cone_add
	    theta = 2 * np.pi * np.random.rand()
	    normals[:,1] = n1_mult * normals[:,1] + beta * np.sin(theta) * U[:,1] + beta * np.cos(theta) * U[:,2]

	    normals = normals / np.tile(np.linalg.norm(normals, axis=0), [3,1])
	    return contacts, normals, num_facets, mu, gamma

class GraspTest():
    def test_init_grasp(self):
	# random grasp
	g1 = np.random.rand(3)
	g2 = np.random.rand(3)
	x = (g1 + g2) / 2
	v = g2 - g1
	width = np.linalg.norm(v)
	v = v / width
	configuration = ParallelJawPtGrasp3D.configuration_from_params(x, v, width)

	# test init
	random_grasp = ParallelJawPtGrasp3D(configuration)
	read_configuration = random_grasp.configuration

	read_g1, read_g2 = random_grasp.endpoints

	# test bad init
	configuration[4] = 1000
	caught_bad_init = False
	try:
	    random_grasp = ParallelJawPtGrasp3D(configuration)
	except:
	    caught_bad_init = True

    def antipodal_grasp_sampler(self):
	of = ObjFile(OBJ_FILENAME)
	sf = SdfFile(SDF_FILENAME)
	mesh = of.read()
	sdf = sf.read()
	obj = GraspableObject3D(sdf, mesh)

	gripper = RobotGripper.load(GRIPPER_NAME)

	ags = AntipodalGraspSampler(gripper, CONFIG)
	grasps = ags.generate_grasps(obj, target_num_grasps=10)
	

        quality_config = GraspQualityConfigFactory.create_config(CONFIG['metrics']['force_closure'])
        quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, quality_config)

	q_to_c = lambda quality: CONFIG['quality_scale']
	
	i = 0
	vis.figure()
	vis.mesh(obj.mesh.trimesh, style='surface')
	for grasp in grasps:
	    success, c = grasp.close_fingers(obj)
            if success:
                c1, c2 = c
                fn_fc = quality_fn(grasp).quality
                true_fc = PointGraspMetrics3D.force_closure(c1, c2, quality_config.friction_coef)
	    #print(grasp)
	    T_obj_world = RigidTransform(from_frame='obj',to_frame='world')
	    color = plt.get_cmap('hsv')(q_to_c(CONFIG['metrics']))[:-1]
	    T_obj_gripper = grasp.gripper_pose(gripper)
            #vis.grasp(grasp,grasp_axis_color=color,endpoint_color=color)
	    i += 1	
	#T_obj_world = RigidTransform(from_frame='obj',to_frame='world')
	vis.show(False)

if __name__ == '__main__':
    grasp = GraspTest()
    grasp.antipodal_grasp_sampler() 
