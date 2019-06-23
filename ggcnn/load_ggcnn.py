import argparse
import logging
import math
import numpy as np
import time
import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from models import get_network
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Load GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Network Name in .models')
    parser.add_argument('--path', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    ggcnn = get_network(args.network)
    model = torch.load(args.path, map_location='cpu')
    #print(model)
    
    depth_im = np.load("/home/silvia/dex-net/data/datasets/fc_4/tensors/depth_images_00000.npz")['arr_0'][50]
    depth_tensor = torch.from_numpy(np.expand_dims(depth_im.reshape((200,200)), 0).astype(np.float32)).reshape((1,1,200,200))
    device = torch.device("cpu")

    xc = depth_tensor.to(device)
    pos_pred, cos_pred, sin_pred = model(xc)

    x_n = xc.detach().numpy().reshape(200,200)
    pos_pred_n = pos_pred.detach().numpy().reshape(200,200)
    cos_pred_n = cos_pred.detach().numpy().reshape(200,200)
    sin_pred_n = sin_pred.detach().numpy().reshape(200,200)
    grasp_angle_n = np.zeros((200,200))
    for x in range(75,125):
        for y in range(75,125):
            grasp_angle_n[x,y] = 0.5 * math.atan2(sin_pred_n[x,y], cos_pred_n[x,y])
    plt.figure(figsize=(14, 4))
    plt.subplot(131)
    plt.imshow(x_n[75:125, 75:125])
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(pos_pred_n[75:125, 75:125])
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(grasp_angle_n[75:125, 75:125])
    # plt.plot([g1p[0], g2p[0]], [g1p[1], g2p[1]], color='firebrick', linewidth=5, linestyle='--')
    plt.colorbar()
    plt.show()



    Dataset = get_dataset(args.dataset)
    print("Before dataset")
    train_dataset = Dataset(args.dataset_path, start=0.0, end=0.1, ds_rotate=args.ds_rotate,
                            random_rotate=True, random_zoom=True,
                            include_depth=args.use_depth, include_rgb=args.use_rgb)
    print("After dataset")
    print("Before dataloader")
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    print("After dataloader")
    # net = torch.load(args.network)
    device = torch.device("cpu")

    i = 0
    for x, y, _, _, _ in train_data:
        xc = x.to(device)
        print(x.shape)
        yc = [yy.to(device) for yy in y]
        policy_start = time.time()
        pos_pred, cos_pred, sin_pred = model(xc)
        print('Planning took %.3f sec' %(time.time() - policy_start))
        y_pos, y_cos, y_sin = yc
        pos_pred_n = pos_pred.detach().numpy().reshape(200,200)
        cos_pred_n = cos_pred.detach().numpy().reshape(200,200)
        sin_pred_n = sin_pred.detach().numpy().reshape(200,200)
        y_pos_n = y_pos.detach().numpy().reshape(200,200)
        y_cos_n = y_cos.detach().numpy().reshape(200,200)
        x_n = xc.detach().numpy().reshape(200,200)

        # plt.figure()
        # plt.subplot(221)
        # plt.imshow(pos_pred_n.reshape(200,200)[75:125])
        # plt.subplot(222)
        # plt.imshow(cos_pred_n.reshape(200,200)[75:125])
        # plt.subplot(223)
        # plt.imshow(y_pos_n.reshape(200,200)[75:125])
        # plt.subplot(224)
        # plt.imshow(x_n.reshape(200,200)[75:125])
        # plt.show()
        grasp_angle_n = np.zeros((200,200))
        for x in range(75,125):
            for y in range(75,125):
                grasp_angle_n[x,y] = 0.5 * math.atan2(sin_pred_n[x,y], cos_pred_n[x,y])

        idx = np.argmax(pos_pred_n)
        x = idx / 200
        y = idx % 200
        axis = np.array([np.cos(grasp_angle_n[x,y]), np.sin(grasp_angle_n[x,y])])
        g1p = [x/4,y/4] - (axis * 10) # start location of grasp jaw 1
        g2p = [x/4,y/4] + (axis * 10) # start location of grasp jaw 2

        # plt.figure()
        # plt.subplot(231)
        # plt.imshow(pos_pred_n[75:125, 75:125])
        # plt.colorbar()
        # plt.subplot(232)
        # plt.imshow(cos_pred_n[75:125, 75:125])
        # plt.colorbar()
        # plt.subplot(233)
        # plt.imshow(sin_pred_n[75:125, 75:125])
        # plt.colorbar()
        # plt.subplot(234)
        # plt.imshow(x_n[75:125, 75:125])
        # plt.colorbar()
        # plt.subplot(235)
        # plt.imshow(grasp_angle_n[75:125, 75:125])
        # plt.colorbar()
        # plt.show()

        plt.figure(figsize=(14, 4))
        plt.subplot(131)
        plt.imshow(x_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(pos_pred_n[75:125, 75:125])
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(grasp_angle_n[75:125, 75:125])
        plt.plot([g1p[0], g2p[0]], [g1p[1], g2p[1]], color='firebrick', linewidth=5, linestyle='--')
        plt.colorbar()
        plt.show()

        i+=1
        if (i > 5):
            break


