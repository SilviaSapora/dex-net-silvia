"""
Visualize a grasp detection dataset
Author: Jeff Mahler
"""
import argparse
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import random
import sys
from perception import DepthImage
from autolab_core import YamlConfig
from dexnet.learning import TensorDataset

# global vars
POSITIVE_CLASS_THRESHOLD = 0.002 # positive class must have a grasp metric greater than this
DATAPOINTS_PER_FILE = 1000 # number of unique datapoints in each file

# templates for filename searching
DEPTH_IM_TEMPLATE = 'depth_images'
START_IND_TEMPLATE = 'start_i'
END_IND_TEMPLATE = 'end_i'
GRASP_TEMPLATE = 'hand_configurations'
METRIC_TEMPLATE = 'robust_ferrari_canny'
LABEL_TEMPLATE = 'grasp_labels'

# plotting params
LINE_WIDTH = 3
FONT_SIZE = 15
MIN_METRIC = 0.0
MAX_METRIC = 0.01

im_final_height = 96
im_final_width = 96
im_crop_height = 192
im_crop_width = 192

def filename_to_file_num(filename):
    """ Extracts the integer file number from a tensor .npz filename. """
    return int(filename[-9:-4])

def global_index_to_file_num(index, datapoints_per_file=DATAPOINTS_PER_FILE):
    """ Returns the file number for the file containing the datapoint at the given index in the full dataset. """
    return index / datapoints_per_file

def global_index_to_file_index(index, datapoints_per_file=DATAPOINTS_PER_FILE):
    """ Returns the index in the file containing the datapoint for the datapoint at the given global index in the full dataset. """
    return index % datapoints_per_file

def load_grasps_for_image(start_ind, end_ind,grasp_tensors):
    """ Loads a set of grasps for an image with a given start and end
    index in the global grasp dataset. """
    grasps = []

    # compute id of file to load and datapoint index in the file
    grasp_start_file_num = global_index_to_file_num(start_ind)
    grasp_end_file_num = global_index_to_file_num(end_ind)
    file_ind = global_index_to_file_index(start_ind)

    # iteratively load grasps from the next file in memory
    num_grasps = end_ind - start_ind
    for i in range(grasp_start_file_num, grasp_end_file_num+1):
        grasp_filename = grasp_tensors[i]
                
        grasp_arr = np.load(grasp_filename)['arr_0']
        
        while file_ind < DATAPOINTS_PER_FILE and len(grasps) < num_grasps:
            grasps.append(grasp_arr[file_ind,:])
            file_ind += 1
        file_ind = 0
    return grasps

def plot_grasp(grasp, metric, color=None):
    """ Plots a given grasp with a given metric score. """
    x = grasp[0:2]    # grasp center pixel
    theta = grasp[3]  # grasp axis angle with the image x axis
    d = grasp[2]      # grasp depth from the camera
    w = grasp[6]      # grasp width, in pixels
    
    v = np.array([np.sin(theta), np.cos(theta)])
    g1 = x + (w/2) * v
    g2 = x - (w/2) * v
    l = np.c_[g1, g2]
    if color is None:
        q = (metric - MIN_METRIC) / (MAX_METRIC - MIN_METRIC)
        c = min(max(q, 0), 1)
        color = plt.cm.RdYlGn(c) 
    plt.plot(l[1,:], l[0,:], linewidth=LINE_WIDTH,
             c=color)

if __name__ == '__main__':
    # parse args
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, default=None, help='path to the detection dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='path to the output dataset')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    config_filename = args.config_filename

    ######################################################################
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/generate_gqcnn_dataset.yaml')
    
    config = YamlConfig(config_filename)

    # set tensor dataset config
    tensor_config = config['tensors']
    tensor_config['fields']['depth_ims_tf_table_96']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table_96']['width'] = im_final_width

    # tensor_dataset = TensorDataset(output_dir, tensor_config)
    tensor_dataset = TensorDataset.open(output_dir)
    tensor_datapoint = tensor_dataset.datapoint_template
    
    ######################################################################

    # open tensor dirs
    if not os.path.exists(dataset_path):
        raise ValueError('Dataset %s not found!' %(dataset_path))

    # create subdirectories
    image_dir = os.path.join(dataset_path, 'images')
    if not os.path.exists(image_dir):
        raise ValueError('Image folder %s not found!' %(image_dir))

    grasp_dir = os.path.join(dataset_path, 'grasps')
    if not os.path.exists(grasp_dir):
        raise ValueError('Grasp folder %s not found!' %(grasp_dir))

    image_tensor_dir = os.path.join(image_dir, 'tensors')
    grasp_tensor_dir = os.path.join(grasp_dir, 'tensors')

    # load image filenames
    image_tensor_filenames = os.listdir(image_tensor_dir)
    image_tensors = [os.path.join(image_tensor_dir,f) for f in image_tensor_filenames if f.startswith(DEPTH_IM_TEMPLATE)]
    start_ind_tensors = [os.path.join(image_tensor_dir,f) for f in image_tensor_filenames if f.startswith(START_IND_TEMPLATE)]
    end_ind_tensors = [os.path.join(image_tensor_dir,f) for f in image_tensor_filenames if f.startswith(END_IND_TEMPLATE)]

    # load grasp filenames
    grasp_tensor_filenames = os.listdir(grasp_tensor_dir)
    grasp_tensors = [os.path.join(grasp_tensor_dir,f) for f in grasp_tensor_filenames if f.startswith(GRASP_TEMPLATE)]

    # sort
    image_tensors.sort(key = filename_to_file_num)
    start_ind_tensors.sort(key = filename_to_file_num)
    end_ind_tensors.sort(key = filename_to_file_num)
    grasp_tensors.sort(key = filename_to_file_num)

    # extract metadata
    num_image_tensors = len(image_tensors)
    num_grasp_tensors = len(grasp_tensors)

    # load and display each image by selecting one uniformly at random
    # CTRL+C to exit
    image_tensor_inds = np.arange(num_image_tensors)

    ########################################################################
    
    for ind in range(200, num_image_tensors):
        print("File: " + str(ind))
        image_arr = np.load(image_tensors[ind])['arr_0']
        start_ind_arr = np.load(start_ind_tensors[ind])['arr_0']
        end_ind_arr = np.load(end_ind_tensors[ind])['arr_0']

        datapoints_in_file = min(len(start_ind_arr), DATAPOINTS_PER_FILE)
        for i in range(datapoints_in_file):
            start_ind = start_ind_arr[i]
            end_ind = end_ind_arr[i]
            
            if end_ind < 3057000:
                continue

            depth_im = DepthImage(image_arr[i,...])
            grasps = load_grasps_for_image(start_ind, end_ind, grasp_tensors)

            # read pixel offsets
            cx = depth_im.center[1]
            cy = depth_im.center[0]

            for g, grasp in enumerate(grasps):
                current_index = start_ind + g
                if (current_index < 3057000):
                    continue
                grasp_x = grasp[1]
                grasp_y = grasp[0]
                grasp_angle = grasp[3]
                
                # center images on the grasp, rotate to image x axis
                dx = cx - grasp_x
                dy = cy - grasp_y
                translation = np.array([dy, dx])

                depth_im_tf_table = depth_im.transform(translation, grasp_angle)

                # crop to image size
                depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)

                # resize to image size
                depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))

                tensor_datapoint['depth_ims_tf_table_96'] = depth_im_tf_table.raw_data
                tensor_dataset.add(tensor_datapoint)

        gc.collect()
    
    tensor_dataset.flush()
    #########################################################################
    # np.random.shuffle(image_tensor_inds)

    # for ind in image_tensor_inds:
    #     print 'Loading image tensor', ind

    #     # load image data
    #     image_arr = np.load(image_tensors[ind])['arr_0']
    #     start_ind_arr = np.load(start_ind_tensors[ind])['arr_0']
    #     end_ind_arr = np.load(end_ind_tensors[ind])['arr_0']

    #     i = np.random.choice(DATAPOINTS_PER_FILE, size=1)[0]
    #     print 'Showing image', i
        
    #     # read datapoint
    #     depth_im = image_arr[i,...]
    #     start_ind = start_ind_arr[i]
    #     end_ind = end_ind_arr[i]
        
    #     # index grasps
    #     grasps, labels, metrics = load_grasps_for_image(start_ind, end_ind,
    #                                                     grasp_tensors,
    #                                                     label_tensors,
    #                                                     metric_tensors)
                
    #     # display
    #     plt.figure(figsize=(10,6))
    #     plt.subplot(1,2,1)
    #     plt.imshow(depth_im[:,:,0], cmap=plt.cm.gray_r)
    #     for grasp, label in zip(grasps, labels):
    #         color = 'r'
    #         if label == 1:
    #             color = 'g'
    #         plot_grasp(grasp, label, color=color)
    #     plt.axis('off')
    #     plt.title('CLASSIFICATION', fontsize=FONT_SIZE)

    #     plt.subplot(1,2,2)
    #     plt.imshow(depth_im[:,:,0], cmap=plt.cm.gray_r)
        
    #     for grasp, metric in zip(grasps, metrics):
    #         plot_grasp(grasp, metric)
    #     plt.axis('off')
    #     plt.title('REGRESSION', fontsize=FONT_SIZE)
    #     plt.show()
