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

if __name__ == '__main__':
	
	fname_known = '/home/silvia/dex-net/policy_res/policy_res_known.txt'
	fname_unknown = '/home/silvia/dex-net/policy_res/policy_res_unknown.txt'
	fname_procedural = '/home/silvia/dex-net/policy_res/policy_res_procedural.txt'
	fname_princeton = '/home/silvia/dex-net/policy_res/policy_res_princeton.txt'
	fname_gqae = '/home/silvia/dex-net/policy_res/policy_res_gqae.txt'
	# fnames = [fname_known, fname_unknown, fname_procedural, fname_princeton]
	# fnames = [fname_unknown]
	fnames = [fname_procedural]
	# fnames = [fname_gqae]
	# model_names = ['GQCNN-RANDOM',
	# 			   'GQCNN-2.0-10', 
	# 			   'GQCNN-2.0-20', 
	# 			   'GQCNN-2.0-50', 
	# 			   'GQCNN-2.0-80', 
	# 			   'GQCNN-2.0']   
	model_names = [
	   'fc_6_angles_fc_5/epoch_10',
	   # 'fc_6_finetuned/epoch_21'
	   ]   
	for fname in fnames:
		f = open(fname)
		num_models = len(model_names)
		total_trials = np.zeros(num_models)
		successes = np.zeros(num_models)
		success_percentage = np.zeros(num_models)
		objects_heatmap = np.zeros((100, num_models))
		while True:
			content = f.readline()
			if content == '':
				break
			res = content.strip().split()
			model_name = res[0]
			obj_number = int(res[1])
			obj_name = res[2]
			try:
				model_num = model_names.index(model_name)
			except:
				continue
			# if quality == "No":
			# 	total_trials[model_num] += 1
			# 	no_grasps[model_num] += 1
			# 	continue
			success = int(res[4])
			if model_num >= 0:
				total_trials[model_num]+= 1
				if success:
					successes[model_num] += 1
					objects_heatmap[obj_number][model_num] += 1
		
		heatmap = []
		obj_fail = []
		for i, res in enumerate(objects_heatmap):
			# if execution failed for all policies don't add
			if all(res == 0) or all(res==3):
				continue
			# if was successful for all policies don't add
			# if all(res[1:6]):
			# 	continue
			heatmap.append(res)
			obj_fail.append(i)

		print(obj_fail)
		print(len(heatmap))
		heatmap = np.swapaxes(heatmap,1,0)
		fig, ax = plt.subplots()
		im = ax.imshow(heatmap)

		# We want to show all ticks...
		ax.set_yticks(np.arange(len(model_names)))
		ax.set_xticks(np.arange(len(obj_fail)))
		# ... and label them with the respective list entries
		ax.set_yticklabels(model_names)
		ax.set_xticklabels(obj_fail)

		# Rotate the tick labels and set their alignment.
		#plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		#         rotation_mode="anchor")

		# Loop over data dimensions and create text annotations.
		#for i in range(len(vegetables)):
		#    for j in range(len(farmers)):
		#        text = ax.text(j, i, harvest[i, j],
		#                       ha="center", va="center", color="w")

		ax.set_title("Object success heatmap")
		fig.tight_layout()
		plt.show()

				
			


