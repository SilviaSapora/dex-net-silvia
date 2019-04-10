import os
import sys
import glob
import h5py
import numpy as np
import trimesh
import time
import math
from scipy import misc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

sys.path.insert(0, '/home/silvia/dex-net/v-rep-grasping/')

import lib
import lib.utils
from lib.config import config_mesh_dir, config_output_collected_dir
import vrep
vrep.simxFinish(-1)
import simulator as SI

sys.path.insert(0, '/home/silvia/dex-net/')
#from stable_pose_grasps import antipodal_grasp_sampler
from store_grasps import SDatabase


IM_HEIGHT_PLANNING = 128
IM_WIDTH_PLANNING = 128
IM_WIDTH_DB = 32
IM_WIDTH_DB = 32

def save_images(rgb_image, depth_image, postfix, save_dir, save_for_planning=False):
    """Saves the queried images to disk."""

    filename_jpg = os.path.join(save_dir, postfix + '.jpg')
    #misc.imsave(name, np.uint8(images[0, :3].transpose(1, 2, 0) * 255))

    misc.imsave(filename_jpg, np.uint8(rgb_image))

    if save_for_planning:
        im = Image.open(filename_jpg)
        filename_png = os.path.join(save_dir, postfix + '.png')
        im.save(filename_png)
        os.remove(filename_jpg)

        filename = os.path.join(save_dir, 'depth_0')
        np.save(filename, np.float32(depth_image.reshape([IM_HEIGHT_PLANNING,IM_WIDTH_PLANNING])), False, True)
        return

    # To write the depth info, save as a 32bit float via numpy
    name = os.path.join(save_dir, postfix)
    np.save(name, np.float32(depth_image.reshape([IM_WIDTH_DB,IM_WIDTH_DB])), False, True)


def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh

def drop_object(sim, initial_pose):
    sim.run_threaded_drop(initial_pose)
    #object_pose = sim.get_object_pose()
    sim.set_object_pose(initial_pose[:3].flatten())

def run_grasps(sim, initial_pose, gripper_poses, object_name, pose_id):
    for count, gripper_pose in enumerate(gripper_poses):
        print('candidate: ', count)

        #work2candidate = gripper_poses[count]

        sim.set_object_pose(initial_pose[:3].flatten())

        # We can randomize the gripper candidate by rotation or translation.
        # Here we let the pose vary +- 3cm along local z, and a random
        # rotation between[0, 360) degress around local z
        #random_pose = lib.utils.randomize_pose(work2candidate,
        #                                       offset_mag=candidate_offset_mag,
        #                                       local_rot=candidate_local_rot)
        #sim.set_gripper_pose(random_pose)
        time.sleep(1)
        # set gripper to exact pose
        collision = sim.set_gripper_pose(gripper_pose)
        sim.set_camera_pose(gripper_pose)

        rgb_image, depth_image = sim.camera_images()

        if rgb_image is None:
                raise Exception('No image returned.')

        if collision:
            print('grasp is colliding')
            time.sleep(1)
            postfix = 'collision_%s_%d_%d' % (object_name, pose_id, count)
            save_images(rgb_image, depth_image, postfix, '/home/silvia/dex-net/v-rep-grasping/output/images')
            continue
        # wait a bit before checking collisions and closing the gripper
        time.sleep(1)

        grasp_res = sim.run_threaded_candidate()

        # SUCCESS
        if (grasp_res == '0'):
            postfix = 'success_%s_%d_%d' % (object_name, pose_id, count)
        else:
            postfix = 'fail_%s_%d_%d' % (object_name, pose_id, count)
        save_images(rgb_image, depth_image, postfix, '/home/silvia/dex-net/v-rep-grasping/output/images', save_as_png=True)

        




    print('Finished Collecting!')

def collect_grasps(sim,
                   mesh_path=None,
                   initial_height=0.5,
                   num_candidates=100,
                   candidate_noise_level=0.1,
                   num_random_per_candidate=1,
                   candidate_offset=-0.07,
                   candidate_offset_mag=0.03,
                   candidate_local_rot=(10, 10, 359),
                   show_pregrasp_pose=False):

    # ----------- CODE TO LOAD MESH IN MESH DIRECTORY -----------
    #if not os.path.exists(config_output_collected_dir):
    #    os.makedirs(config_output_collected_dir)

    #mesh_name = mesh_path.split(os.path.sep)[-1]

    # Load the mesh from file here, so we can generate grasp candidates
    # and access object-specifsc properties like inertia.
    #mesh = load_mesh(mesh_path)
    # ----------- END CODE TO LOAD MESH IN MESH DIRECTORY ----------
    
    for i in range(5):
        print('OBJECT ', i)
        object_name = 'example' + str(i)
        mesh_path = '/home/silvia/dex-net/.dexnet/' + object_name + '_proc.obj'
        # CREATE MESH
        #sim.create_object(mesh_path)
        #db.database_save(object_name, mesh_path)
        #continue
        # SAVE OBJECTS
        # obj_names = ['obj0', 'obj1', 'obj2', 'obj3', 'obj4']
        # for obj_n, obj_name in enumerate(obj_names):
        #     mesh_path = '/home/silvia/dex-net/generated_shapes/example' + str() + '.obj'
        #     sim.save_object(mesh_path, obj_name)
        # LOAD MESH
        mesh = load_mesh(mesh_path)
        mass = mesh.mass_properties['mass'] * 10
        com = mesh.mass_properties['center_mass']
        inertia = mesh.mass_properties['inertia'] * 5
        sim.load_object(mesh_path, com, mass, inertia.flatten())
        # open database
        db = SDatabase('/home/silvia/dex-net/silvia.hdf5', 'main')

        for pose_id in range(5):
            print("pose id: ", pose_id)
            # ----------- GET POSES AND GRASPS FOR GIVEN OBJECT AND POSE -----------
            initial_pose = db.get_stable_pose(object_name, pose_id)
            grasps, gripper_poses = db.stable_pose_grasps(object_name, pose_id, max_grasps=10, visualize=False)
            if grasps == None:
                continue
            # ----------------------------------------------------------------------

            drop_object(sim, initial_pose)
            run_grasps(sim, initial_pose, gripper_poses[:10], object_name, pose_id)

def save_camera_images(sim):
    rgb_image, depth_image = sim.camera_images()
    postfix = 'color_0'
    save_images(rgb_image, depth_image, postfix, '/home/silvia/dex-net/v-rep-grasping/output/images', save_for_planning=True)


if __name__ == '__main__':

    # Use the spawn_headless = False / True flag to view with GUI or not
    spawn_params = {'port': 19997,
                    'ip': '127.0.0.1',
                    'vrep_path': None,
                    'scene_path': None,
                    'exit_on_stop': True,
                    'spawn_headless': False,
                    'spawn_new_console': True}

    # Sample way for calling VREP on windows by specifying full path:
    # spawn_params['vrep_path'] = 'C:\\Program Files\\V-REP3\\V-REP_PRO_EDU\\vrep.exe'
    
    sim = SI.SimulatorInterface(**spawn_params)
    #sim = []

    if len(sys.argv) == 1:
        collect_grasps(sim)
        # save_camera_images(sim)

    else:
        spawn_params['port'] = int(sys.argv[1])

        # List of meshes we should run are stored in a file,
        mesh_list_file = sys.argv[2]
        with open(mesh_list_file, 'r') as f:
            while True:
                mesh_path = f.readline().rstrip()

                if mesh_path == '':
                    break
                collect_grasps(sim, mesh_path, num_candidates=1000)
