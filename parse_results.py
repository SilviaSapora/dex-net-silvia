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
	fname_gqae_3 = '/home/silvia/dex-net/policy_res/gqae_res.txt'
	fname_test = '/home/silvia/dex-net/policy_res/test.txt'
	fname_princeton_plus_procedural = '/home/silvia/dex-net/policy_res/princeton_plus_procedural.txt'
	fname_gqae_fixed1 = '/home/silvia/dex-net/policy_res/gqae_res_fixed_known.txt'
	fname_gqae_fixed2 = '/home/silvia/dex-net/policy_res/gqae_res_fixed_unknown.txt'
	fname_gqae_fixed3 = '/home/silvia/dex-net/policy_res/gqae_res_fixed_procedural.txt'
	fname_gqae_fixed4 = '/home/silvia/dex-net/policy_res/gqae_res_fixed_princeton.txt'
	fnames = [
			  # fname_known, 
			  # fname_unknown, 
			  # fname_procedural, 
			  # fname_princeton,
			  fname_gqae_fixed1,
			  fname_gqae_fixed2,
			  fname_gqae_fixed3,
			  fname_gqae_fixed4
			  # fname_princeton_plus_procedural, 
			  # fname_test,
			  # fname_gqae, 
			  # fname_gqae_3
			  ]
	# fnames = [fname_procedural]
	# fnames = [fname_gqae]
	model_names = [
				   # 'GQCNN-2.0-10',             # 0 
				   # 'GQCNN-2.0-20',             # 1
				   # 'GQCNN-2.0-50',             # 2
				   # 'GQCNN-2.0-80',             # 3
				   # 'GQCNN-2.0',                # 4
				   # 'GQCNN-RANDOM',             # 5
				   # 'GQCNN-2.0-color',          # 6
				   # 'GQCNN-2.0d-96-fixed',      # 7
				   # 'GQCNN-50-procedural',      # 8
				   # 'fc_6_BCE/epoch_27',        # 9
				   # 'GQCNN-2.0-5333-32',        # 10
       #             'GQCNN-2.0d-ferrari',       # 11
       #             'GQCNN-2.0d-force_closure', # 12
                   # 'epoch_13_iou_0.00',        # 13
                   # 'epoch_06_v1_iou_0.00',     # 14
                   # 'epoch_06_v2_iou_0.00',     # 15
                   'fc_6_angles_fc_5/epoch_10',# 16
                   # 'fc_6_angles_fc_5/epoch_20',# 17
                   # 'fc_6_finetuned/epoch_21',  # 18
                   # 'fc_6_0default_angles/epoch_27']# 19
                   ]
	for fname in fnames:
		f = open(fname)
		num_models = len(model_names)
		total_trials = np.zeros(num_models)
		failures = np.zeros(num_models)
		successes = np.zeros(num_models)
		success_percentage = np.zeros(num_models)
		accuracy = np.zeros(num_models)
		accuracy_yes = np.zeros(num_models)
		accuracy_no = np.zeros(num_models)
		mse = np.zeros(num_models)
		rob_rate = np.zeros(num_models)
		no_grasps = np.zeros(num_models)
		while True:
			content = f.readline()
			if content == '':
				break
			res = content.strip().split()
			model_name = res[0]
			obj_number = int(res[1])
			obj_name = res[2]
			quality = res[3]
			try:
				model_num = model_names.index(model_name)
			except:
				continue
			if quality == "No":
				total_trials[model_num]+= 1
				no_grasps[model_num] += 1
				continue
			success = int(res[4])
			quality = float(quality)
			if model_num >= 0:
				total_trials[model_num]+= 1
				if (quality) >= 0.5:
					rob_rate[model_num] += 1
				if success:
					successes[model_num] += 1
					if (quality) >= 0.5:
						accuracy[model_num] += 1
						accuracy_yes[model_num] += 1
					mse[model_num] += (quality-1) ** 2
				else:
					failures[model_num] += 1
					if (quality) < 0.5:
						accuracy[model_num] += 1
						accuracy_no[model_num] += 1
					mse[model_num] += (quality) ** 2
		
		for m, _ in enumerate(model_names):
			if total_trials[m] == 0:
				continue
			if m == 'GQCNN-2.0-color':
				successes[m] = successes[m] / float(total_trials[m])
				failures[m] = (failures[m] - np.max(no_grasps)) / float(total_trials[m])
				no_grasps[m] = no_grasps[m] / float(total_trials[m])
				accuracy[m] = accuracy[m] / float(total_trials[m])
			else:
				accuracy_yes[m] = accuracy_yes[m] / float(rob_rate[m])
				accuracy_no[m] = accuracy_no[m] / float(failures[m])
				successes[m] = successes[m] / float(total_trials[m])
				failures[m] = failures[m] / float(total_trials[m])
				no_grasps[m] = no_grasps[m] / float(total_trials[m])
				accuracy[m] = accuracy[m] / float(total_trials[m])
				mse[m] = mse[m] / float(total_trials[m])
				rob_rate[m] = rob_rate[m] / float(total_trials[m])

		print("")
		print("")
		print(fname)
		print(model_names)
		print("successes")
		print(successes)
		# print("failures")
		# print(failures)
		# print("no grasps")
		# print(no_grasps)
		print("accuracy")
		print(accuracy)
		print("robusteness")
		print(rob_rate)
		print("accuracy_yes")
		print(accuracy_yes)
		# print("mse")
		# print(mse)
		print("total trials")
		print(total_trials)

		# print('')
		# print('robust ferrari canny VS ferrari canny VS force closure')
		# print('GQCNN-2.0: s = ' + str(successes[4]) 
		# 			 + '. f = ' + str(failures[4]) 
		# 			 + '. acc = ' + str(accuracy[4])
		# 			 + '. mse = ' + str(mse[4])
		# 			 + '. acc yes = ' + str(accuracy_yes[4])
		# 			 + '. robusteness rate = ' + str(rob_rate[4])
		# 			 + '. tot = ' + str(total_trials[4]))

		# print('GQCNN-2.0d-ferrari: s = ' + str(successes[11]) 
		# 					  + '. f = ' + str(failures[11]) 
		# 					  + '. acc = ' + str(accuracy[11])
		# 					  + '. mse = ' + str(mse[11])
		# 					  + '. acc yes = ' + str(accuracy_yes[11])
		# 			          + '. robusteness rate = ' + str(rob_rate[11])
		# 					  + '. tot = ' + str(total_trials[11]))

		# print('GQCNN-2.0d-force_closure: s = ' + str(successes[12]) 
		# 							+ '. f = ' + str(failures[12]) 
		# 							+ '. acc = ' + str(accuracy[12])
		# 							+ '. mse = ' + str(mse[12])
		# 							+ '. acc yes = ' + str(accuracy_yes[12])
		# 			 				+ '. robusteness rate = ' + str(rob_rate[12])
		# 							+ '. tot = ' + str(total_trials[12]))


		# print('')
		# print('dexnet hyper convs VS my hyper convs VS 96x96')
		# print('GQCNN-2.0: s = ' + str(successes[4]) 
		# 					 + '. f = ' + str(failures[4]) 
		# 					 + '. acc = ' + str(accuracy[4])
		# 					 + '. acc yes = ' + str(accuracy_yes[4])
		# 			         + '. robusteness rate = ' + str(rob_rate[4])
		# 					 + '. tot = ' + str(total_trials[4]))

		# print('GQCNN-2.0-5333-32: s = ' + str(successes[10]) 
		# 			 		 + '. f = ' + str(failures[10]) 
		# 			 		 + '. acc = ' + str(accuracy[10])
		# 			 		 + '. acc yes = ' + str(accuracy_yes[10])
		# 			         + '. robusteness rate = ' + str(rob_rate[10])
		# 			 		 + '. tot = ' + str(total_trials[10]))

		# print('GQCNN-2.0-96x96: s = ' + str(successes[7]) 
		# 			 		 + '. f = ' + str(failures[7]) 
		# 			 		 + '. acc = ' + str(accuracy[7])
		# 			 		 + '. acc yes = ' + str(accuracy_yes[7])
		# 			         + '. robusteness rate = ' + str(rob_rate[7])
		# 			 		 + '. tot = ' + str(total_trials[7]))


		# plt.figure(figsize=(14, 4))
		# plt.subplot(131)
		# plt.plot([10,20,50,80,100], successes[:5], color='green')
		# plt.plot([10,100], [successes[5], successes[5]], color='green', linestyle='dashed')
		# plt.plot([10,100], [successes[6], successes[6]], color='pink', linestyle='dashed')
		# plt.plot([10,100], [successes[7], successes[7]], color='black', linestyle='dashed')
		# plt.plot([10,100], [successes[8], successes[8]], color='cyan', linestyle='dashed')
		# plt.title('Success')

		# plt.subplot(132)
		# plt.plot([10,20,50,80,100], 1-accuracy_yes[:5], color='green')
		# plt.plot([10,20,50,80,100], 1-accuracy_no[:5], color='red')
		# plt.title('False Positives/False Negatives')
		# plt.plot([10,100], [failures[5], failures[5]], color='red', linestyle='dashed')
		# # plt.plot([10,100], [failures[6], failures[6]], color='pink', linestyle='dashed')
		# # plt.plot([10,100], [failures[7], failures[7]], color='black', linestyle='dashed')
		# # plt.plot([10,100], [failures[8], failures[8]], color='cyan', linestyle='dashed')
		# plt.title('Failures')

		# plt.subplot(223)
		# plt.plot([10,20,50,80,100], no_grasps[:5], color='red')
		# plt.plot([10,100], [no_grasps[5], no_grasps[5]], color='red', linestyle='dashed')
		# plt.plot([10,100], [no_grasps[6], no_grasps[6]], color='pink', linestyle='dashed')
		# plt.plot([10,100], [no_grasps[7], no_grasps[7]], color='black', linestyle='dashed')
		# plt.plot([10,100], [no_grasps[8], no_grasps[8]], color='cyan', linestyle='dashed')
		# plt.title('No Grasps')

		# plt.subplot(133)
		# plt.plot([10,20,50,80,100], accuracy[:5], color='red')
		# plt.plot([10,20,50,80,100], mse[:5], color='red')
		# plt.plot([10,100], [accuracy[6], accuracy[6]], color='pink', linestyle='dashed')
		# plt.plot([10,100], [accuracy[7], accuracy[7]], color='black', linestyle='dashed')
		# plt.plot([10,100], [accuracy[8], accuracy[8]], color='cyan', linestyle='dashed')
		# plt.title('MSE')
		# plt.show()
			


