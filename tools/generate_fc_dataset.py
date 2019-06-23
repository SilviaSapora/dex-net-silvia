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
import time
import math
from perception import DepthImage, BinaryImage, CameraIntrinsics, ColorImage, RgbdImage
from autolab_core import YamlConfig, Point
from dexnet.learning import TensorDataset
from gqcnn.grasping import GraspQualityFunctionFactory, RgbdImageState, Grasp2D

# global vars
POSITIVE_CLASS_THRESHOLD = 0.002 # positive class must have a grasp metric greater than this
DATAPOINTS_PER_FILE = 1000 # number of unique datapoints in each file

# templates for filename searching
DEPTH_IM_TEMPLATE = 'depth_images'
BINARY_IM_TEMPLATE = 'binary_images'
START_IND_TEMPLATE = 'start_i'
END_IND_TEMPLATE = 'end_i'
GRASP_TEMPLATE = 'hand_configurations'
METRIC_TEMPLATE = 'robust_ferrari_canny'
LABEL_TEMPLATE = 'grasp_labels'
METRIC_TEMPLATE = 'robust_ferrari_canny'

# plotting params
LINE_WIDTH = 3
FONT_SIZE = 15
MIN_METRIC = 0.0
MAX_METRIC = 0.01

def filename_to_file_num(filename):
    """ Extracts the integer file number from a tensor .npz filename. """
    return int(filename[-9:-4])

def global_index_to_file_num(index, datapoints_per_file=DATAPOINTS_PER_FILE):
    """ Returns the file number for the file containing the datapoint at the given index in the full dataset. """
    return index / datapoints_per_file

def global_index_to_file_index(index, datapoints_per_file=DATAPOINTS_PER_FILE):
    """ Returns the index in the file containing the datapoint for the datapoint at the given global index in the full dataset. """
    return index % datapoints_per_file

def load_grasps_for_image(start_ind, end_ind,grasp_tensors, metric_tensors):
    """ Loads a set of grasps for an image with a given start and end
    index in the global grasp dataset. """
    grasps = []
    metrics = []

    # compute id of file to load and datapoint index in the file
    grasp_start_file_num = global_index_to_file_num(start_ind)
    grasp_end_file_num = global_index_to_file_num(end_ind)
    file_ind = global_index_to_file_index(start_ind)

    # iteratively load grasps from the next file in memory
    num_grasps = end_ind - start_ind
    for i in range(grasp_start_file_num, grasp_end_file_num+1):
        grasp_filename = grasp_tensors[i]
        metric_filename = metric_tensors[i]
                
        grasp_arr = np.load(grasp_filename)['arr_0']
        metric_arr = np.load(metric_filename)['arr_0']
        
        while file_ind < DATAPOINTS_PER_FILE and len(grasps) < num_grasps:
            grasps.append(grasp_arr[file_ind,:])
            metrics.append(metric_arr[file_ind])
            file_ind += 1
        file_ind = 0
    return grasps, metrics

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
    # tensor_config['fields']['quality_ims_tf']['height'] = 200
    # tensor_config['fields']['quality_ims_tf']['width'] = 200
    tensor_config['fields']['grasp_angle_ims_tf']['height'] = 200
    tensor_config['fields']['grasp_angle_ims_tf']['width'] = 200
    # tensor_config['fields']['depth_images']['width'] = 200
    # tensor_config['fields']['depth_images']['height'] = 200

    tensor_dataset = TensorDataset(output_dir, tensor_config)
    # tensor_dataset = TensorDataset.open(output_dir)
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
    binary_image_tensors = [os.path.join(image_tensor_dir,f) for f in image_tensor_filenames if f.startswith(BINARY_IM_TEMPLATE)]
    start_ind_tensors = [os.path.join(image_tensor_dir,f) for f in image_tensor_filenames if f.startswith(START_IND_TEMPLATE)]
    end_ind_tensors = [os.path.join(image_tensor_dir,f) for f in image_tensor_filenames if f.startswith(END_IND_TEMPLATE)]

    # load grasp filenames
    grasp_tensor_filenames = os.listdir(grasp_tensor_dir)
    grasp_tensors = [os.path.join(grasp_tensor_dir,f) for f in grasp_tensor_filenames if f.startswith(GRASP_TEMPLATE)]
    metric_tensors = [os.path.join(grasp_tensor_dir,f) for f in grasp_tensor_filenames if f.startswith(METRIC_TEMPLATE)]

    # sort
    image_tensors.sort(key = filename_to_file_num)
    binary_image_tensors.sort(key = filename_to_file_num)
    start_ind_tensors.sort(key = filename_to_file_num)
    end_ind_tensors.sort(key = filename_to_file_num)
    grasp_tensors.sort(key = filename_to_file_num)
    metric_tensors.sort(key = filename_to_file_num)

    # extract metadata
    num_image_tensors = len(image_tensors)
    num_grasp_tensors = len(grasp_tensors)

    # load and display each image by selecting one uniformly at random
    # CTRL+C to exit
    image_tensor_inds = np.arange(num_image_tensors)

    ########################################################################
    camera_intr_filename = "/home/silvia/dex-net/planning/primesense_overhead_ir.intr"
    camera_intr = CameraIntrinsics.load(camera_intr_filename)
    color_im = ColorImage(np.zeros([400, 400, 3]).astype(np.uint8))
    grasp_quality_fn = GraspQualityFunctionFactory.quality_function('gqcnn', config['metric'])


    for ind in range(num_image_tensors):
        print("File: " + str(ind))
        image_arr = np.load(image_tensors[ind])['arr_0']
        binary_arr = np.load(binary_image_tensors[ind])['arr_0']
        start_ind_arr = np.load(start_ind_tensors[ind])['arr_0']
        end_ind_arr = np.load(end_ind_tensors[ind])['arr_0']

        datapoints_in_file = min(len(start_ind_arr), DATAPOINTS_PER_FILE)
        for i in range(datapoints_in_file):
            start_ind = start_ind_arr[i]
            end_ind = end_ind_arr[i]

            # depth_im = DepthImage(image_arr[i,...])
            binary_im = BinaryImage(binary_arr[i,...])
            grasps, metrics = load_grasps_for_image(start_ind, end_ind, grasp_tensors, metric_tensors)
            
            # # read pixel offsets
            # cx = depth_im.center[1]
            # cy = depth_im.center[0]

            # for g, grasp in enumerate(grasps):
            #     current_index = start_ind + g
            #     grasp_x = grasp[1]
            #     grasp_y = grasp[0]
            #     grasp_angle = grasp[3]
                
            #     # center images on the grasp, rotate to image x axis
            #     dx = cx - grasp_x
            #     dy = cy - grasp_y
            #     translation = np.array([dy, dx])

            #     depth_im_tf_table = depth_im.transform(translation, grasp_angle)

            #     # crop to image size
            #     depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)

            #     # resize to image size
            #     depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))

            #     tensor_datapoint['depth_ims_tf_table_96'] = depth_im_tf_table.raw_data
            #     tensor_dataset.add(tensor_datapoint)
            # fc_im_start = time.time()
            
            # depth_im = depth_im.resize((200, 200))
            # binary_im = binary_im.resize((200, 200))
            # grasp_angle_image = binary_im.raw_data/255 * -10.0
            # object_pixels_x, object_pixels_y = binary_im.raw_data.nonzero()[:2]

            # plt.figure(figsize=(10,6))
            # plt.subplot(1,2,1)
            # plt.imshow(depth_im.raw_data[:,:,0], cmap=plt.cm.gray_r)

            # plt.axis('off')
            # plt.title('DEPTH', fontsize=FONT_SIZE)

            # plt.subplot(1,2,2)
            # plt.imshow(binary_im.raw_data[:,:,0], cmap=plt.cm.gray_r)
            
            # plt.axis('off')
            # plt.title('BINARY', fontsize=FONT_SIZE)
            # plt.show()

            # cx = depth_im.center[1]
            # cy = depth_im.center[0]
            # print("CENTER PIXELS")

            # rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
            # state = RgbdImageState(rgbd_im, camera_intr)

            # quality_image = np.full([200,200,1], 0.0)
            grasp_angle_image = np.full([200,200,1], 0.0)
            # count = np.zeros([200,200])

            metrics_grasps = sorted(zip(metrics,grasps), key=lambda x: x[0])

            for metric, grasp in metrics_grasps:
                grasp_x = int(grasp[0] / 2) % 200
                grasp_y = int(grasp[1] / 2) % 200
                grasp_angle = grasp[3]
                # quality_image[grasp_x-2:grasp_x+1, grasp_y-2:grasp_y+1, 0] += ((metric / 0.002) % 1.0) / 2
                # grasp_angle_image[grasp_x-2:grasp_x+1, grasp_y-2:grasp_y+1, 0] = grasp_angle
                if (metric >= 0.002):
                    # quality_image[grasp_x, grasp_y, 0] = 1
                    # quality_image[grasp_x, grasp_y, 0] += (metric / 0.002) % 1.0
                    # grasp_angle_image[grasp_x, grasp_y, 0] = grasp_angle
                    if np.cos(grasp_angle) < 0:
                            grasp_angle -= math.pi
                    if grasp_angle > 3.0/2.0 * math.pi:
                            grasp_angle -= 2 * math.pi
                    grasp_angle_image[grasp_x-1:grasp_x+2, grasp_y-1:grasp_y+2, 0] = grasp_angle
                    # quality_image[grasp_x, grasp_y, 0] = 1
                    # quality_image[grasp_x-1:grasp_x+2, grasp_y-1:grasp_y+2, 0] = 1
                # else:
                    # grasp_angle_image[grasp_x-1:grasp_x+2, grasp_y-1:grasp_y+2, 0] = grasp_angle
                    # quality_image[grasp_x-1:grasp_x+2, grasp_y-1:grasp_y+2, 0] = grasp_angle
                    # quality_image[grasp_x, grasp_y, 0] = 1
            # for x,y in zip(object_pixels_x, object_pixels_y):
            #     if quality_image[x, y, 0] != 0:
            #         continue
            #     grasps = []
            #     translation = np.array([x*2, y*2])
            #     grasp_center_pt = Point(translation)
                
            #     for grasp_angle in range(-90,81,10):
            #         #depth_im_tf_table = depth_im.transform(translation, grasp_angle)
            #         #depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)
            #         #depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))
            #         grasps.append(Grasp2D(grasp_center_pt, math.radians(grasp_angle), depth_im.raw_data[x*2,y*2,0], camera_intr=camera_intr))
                
            #     q_values = grasp_quality_fn(state, grasps, params=config)
            #     index = np.argmax(q_values)
            #     quality_image[x,y,0] = q_values[index]
            #     grasp_angle_image[x,y,0] = grasps[index].angle
            
            # tensor_datapoint['quality_ims_tf'] = quality_image
            # tensor_datapoint['depth_images'] = depth_im.raw_data
            tensor_datapoint['grasp_angle_ims_tf'] = grasp_angle_image
            tensor_dataset.add(tensor_datapoint)
            
            # logging.info('Image creation took %.3f sec' %(time.time() - fc_im_start))


            # plt.figure()
            # plt.subplot(131)
            # plt.title('DEPTH', fontsize=FONT_SIZE)
            # depth_im = depth_im.resize((200,200))
            # plt.imshow(depth_im.raw_data[70:130,70:130,0])

            # plt.subplot(132)
            # plt.title('QUALITY', fontsize=FONT_SIZE)
            # quality_image = quality_image[70:130,70:130,:]
            # plt.imshow(quality_image[:,:,0])

            # plt.subplot(133)
            # plt.title('ANGLE', fontsize=FONT_SIZE)
            # grasp_angle_image = grasp_angle_image[70:130,70:130,:]
            # plt.imshow(grasp_angle_image[:,:,0])
            
            # plt.show()
            # break
        gc.collect()
        # break
    
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
