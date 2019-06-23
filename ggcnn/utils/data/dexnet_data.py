import numpy as np
import glob
import torch
import os
import torch.utils.data

from sklearn.preprocessing import normalize
from skimage.transform import rotate, resize


import random


class DexNetDataset(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, dataset_path, start=0.0, end=1.0, ds_rotate=0, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = 200
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.start = start
        self.end = end

        qualityf = glob.glob(os.path.join(dataset_path, 'tensors', 'quality_*'))
        anglef = glob.glob(os.path.join(dataset_path, 'tensors', 'grasp_angle_*'))
        depthf = glob.glob(os.path.join(dataset_path, 'tensors', 'depth_images_*'))
        # depthf = glob.glob(os.path.join('/home/silvia/dex-net/data/datasets/3dnet_kit_detection_08_11_17/images', 'tensors', 'depth_images_*'))
        qualityf.sort()
        anglef.sort()
        depthf.sort()
        
        file_num = len(qualityf)
        if file_num == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        #print("end" + str(end))
        #print("file num")
        #print(file_num)
        #print("file end * start")
        #print(file_num*start)
        #print("file end * end")
        #print(file_num*end)
        self.depth_files = depthf[int(file_num*start):int(file_num*end)]
        self.quality_files = qualityf[int(file_num*start):int(file_num*end)]
        self.grasp_angle_files = anglef[int(file_num*start):int(file_num*end)]
        #print("len depthf")
        #print(len(depthf))
        #print("len self depth files")
        #print(len(self.depth_files))
        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_depth(self, idx, rot=0, zoom=1.0):
        # depth_im = DepthImage(image_arr[i,...])
        # depth_im.resize((200, 200))
        depth_im = np.load(self.depth_files[idx/1000])['arr_0'][idx%1000]
        depth_im = depth_im.reshape((self.output_size, self.output_size))
        # depth_im = resize(depth_im, (self.output_size, self.output_size), preserve_range=True, anti_aliasing=True).astype(depth_im.dtype)
        return depth_im
        # depth_np = normalize(depth_np[:,np.newaxis], axis=0).ravel()
        # return depth_np
    
    def get_quality(self, idx, rot=0, zoom=1.0):
        return (np.load(self.quality_files[idx/1000])['arr_0'][idx%1000]).reshape((self.output_size, self.output_size))

    def get_grasp_angle(self, idx, rot=0, zoom=1.0):
        return (np.load(self.grasp_angle_files[idx/1000])['arr_0'][idx%1000]).reshape((self.output_size, self.output_size))

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        raise NotImplementedError()

    def __getitem__(self, idx, rot=0, zoom_factor=1.0):
        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        quality_img = self.get_quality(idx, rot, zoom_factor)
        grasp_angle_img = self.get_grasp_angle(idx, rot, zoom_factor)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(quality_img)
        ignore_idx = np.where(quality_img == -1)
        cos_n = np.cos(2*grasp_angle_img)
        sin_n = np.sin(2*grasp_angle_img)
        cos_n[ignore_idx] = -10
        sin_n[ignore_idx] = -10
        cos = self.numpy_to_torch(cos_n)
        sin = self.numpy_to_torch(sin_n)

        return x, (pos, cos, sin), idx, rot, zoom_factor

    def __len__(self):
        #TODO: FIX
        if self.end != 1.0:
            return len(self.depth_files) * 1000
        else:
            length = (len(self.depth_files)-1) * 1000 + 750
            #print(length)
            return length
