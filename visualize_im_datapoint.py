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
import argparse

from autolab_core import RigidTransform, YamlConfig, BagOfPoints, PointCloud

if __name__ == '__main__':
	# parser = argparse.ArgumentParser(description='Vis file')
    # parser.add_argument('file_path', type=str, default=None, help='name of file to visualize')
	# args = parser.parse_args()
	# path = args.file_path
	path1 = '/home/silvia/dex-net/data/datasets/fc_6/tensors/depth_images_00000.npz'
	path2 = '/home/silvia/dex-net/data/datasets/fc_6/tensors/quality_ims_tf_00000.npz'
	path3 = '/home/silvia/dex-net/data/datasets/fc_6/tensors/grasp_angle_ims_tf_00000.npz'
	f1 = np.load(path1)['arr_0']
	f2 = np.load(path2)['arr_0']
	f3 = np.load(path3)['arr_0']
	print(np.unique(f3))
	print(np.max(f3))
	print(np.min(f3))
	for i in range(50):
		ff1 = (f1[i]).astype(np.float32)
		ff2 = (f2[i]).astype(np.float32)
		ff3 = (f3[i]).astype(np.float32)
		#ff2 = np.swapaxes(ff2,0,1)
		#ff3 = np.swapaxes(ff3,0,1)
		#print(np.sum(ff2==1))
		plt.figure()
		plt.subplot(131)
		plt.title("DEPTH")
		plt.imshow(ff1[70:130,70:130,0])
		plt.colorbar()
		plt.subplot(132)
		plt.title("QUALITY")
		plt.imshow(ff2[70:130,70:130,0])
		plt.colorbar()
		plt.subplot(133)
		plt.title("ANGLE")
		plt.imshow(ff3[70:130,70:130,0])
		plt.colorbar()
		plt.show()
			


