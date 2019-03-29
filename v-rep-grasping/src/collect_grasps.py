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

# sys.path.append('/..')
sys.path.insert(0, '/home/silvia/dex-net/v-rep-grasping/')

import lib
import lib.utils
from lib.config import config_mesh_dir, config_output_collected_dir
import vrep
vrep.simxFinish(-1)
import simulator as SI

sys.path.append('../dex-net')
from stable_pose_grasps import antipodal_grasp_sampler
from store_grasps import SDatabase


def save_images(rgb_image, depth_image, postfix, save_dir):
    """Saves the queried images to disk."""

    name = os.path.join(save_dir, postfix + '_rgb.jpg')
    #misc.imsave(name, np.uint8(images[0, :3].transpose(1, 2, 0) * 255))

    misc.imsave(name, np.uint8(rgb_image))

    # To write the depth info, save as a 16bit float via numpy
    name = os.path.join(save_dir, postfix + '_depth')
    #np.save(name, np.float16(images[0, 3]), False, True)
    np.save(name, np.float16(depth_image), False, True)


def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh


def plot_grasp(mesh_path, pre_or_post_grasp):
    """Plots the contact positions of a grasp & object."""

    frame_work2obj = pre_or_post_grasp['frame_work2obj']
    frame_work2obj = lib.utils.format_htmatrix(frame_work2obj)

    axis = lib.utils.plot_mesh(mesh_path, frame_work2obj, axis=None)
    axis.scatter(*pre_or_post_grasp['work2contact0'], c='r', marker='o', s=75)
    axis.scatter(*pre_or_post_grasp['work2contact1'], c='g', marker='o', s=75)
    axis.scatter(*pre_or_post_grasp['work2contact2'], c='b', marker='o', s=75)
    return axis


def plot_mesh_with_normals(mesh, matrices, direction_vec, axis=None):
    """Visualize where we will sample grasp candidates from

    Parameters
    ----------
    mesh_path : path to a given mesh
    workspace2obj : 4x4 transform matrix from the workspace to object
    axis : (optional) a matplotlib axis for plotting a figure
    """

    if isinstance(direction_vec, list):
        dvec = np.atleast_2d(direction_vec).T
    elif isinstance(direction_vec, np.ndarray) and direction_vec.ndim == 1:
        dvec = np.atleast_2d(direction_vec).T
    else:
        dvec = direction_vec

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    # Construct a 3D mesh via matplotlibs 'PolyCollection'
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5, 0.5, 1])
    axis.add_collection3d(poly)

    axis = lib.utils.plot_equal_aspect(mesh.vertices, axis)

    for i in range(0, len(matrices)):
        transform = lib.utils.format_htmatrix(matrices[i])

        # We'll find the direction by finding the vector between two points
        gripper_point = np.hstack([matrices[i, 3], matrices[i, 7], matrices[i, 11]])
        gripper_point = np.atleast_2d(gripper_point)

        direction = np.dot(transform[:3, :3], dvec)
        direction = np.atleast_2d(direction).T

        a = np.hstack([gripper_point, -direction]).flatten()
        axis.quiver(*a, color='k', length=0.1)

        axis.scatter(*gripper_point.flatten(), c='b', marker='o', s=10)

    return axis

def plot_mesh_with_points(mesh, points, direction_vec, axis=None):
    """Visualize where we will sample grasp candidates from

    Parameters
    ----------
    mesh_path : path to a given mesh
    workspace2obj : 4x4 transform matrix from the workspace to object
    axis : (optional) a matplotlib axis for plotting a figure
    """

    if isinstance(direction_vec, list):
        dvec = np.atleast_2d(direction_vec).T
    elif isinstance(direction_vec, np.ndarray) and direction_vec.ndim == 1:
        dvec = np.atleast_2d(direction_vec).T
    else:
        dvec = direction_vec

    if axis is None:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.autoscale(False)

    # Construct a 3D mesh via matplotlibs 'PolyCollection'
    poly = Poly3DCollection(mesh.triangles, linewidths=0.05, alpha=0.25)
    poly.set_facecolor([0.5, 0.5, 1])
    axis.add_collection3d(poly)

    axis = lib.utils.plot_equal_aspect(mesh.vertices, axis)

    for c1, c2 in points:
        # We'll find the direction by finding the vector between two points
        gripper_point = c1

        #direction = c2-c1

        a = np.array([c1, c2])
        #axis.quiver(*a, color='k', length=0.5)
        axis.plot([c1[0], c2[0]],[c1[1], c2[1]],[c1[2], c2[2]], color='k')

        axis.scatter(*gripper_point.flatten(), c='b', marker='o', s=10)

    return axis
"""
def generate_candidates(mesh, num_samples=10, noise_level=0.05,
                        gripper_offset=-0.1, augment=True):
    #Generates grasp candidates via surface normals of the object.

    # Defines the up-vector for the workspace frame
    up_vector = np.asarray([0, 0, -1])

    points, face_idx = trimesh.sample.sample_surface_even(mesh, num_samples)

    matrices = []
    for p, face in zip(points, face_idx):
        normal = lib.utils.normalize_vector(mesh.triangles_cross[face])
        # print(normal, mesh.triangles_cross[face])

        # Add random noise to the surface normals, centered around 0
        if augment is True:
            normal += np.random.uniform(-noise_level, noise_level)
            normal = lib.utils.normalize_vector(normal)

        # Since we need to set a pose for the gripper, we need to calculate the
        # rotation matrix from a given surface normal
        matrix = lib.utils.get_rot_mat(up_vector, normal)
        matrix[:3, 3] = p

        # Calculate an offset for the gripper from the object.
        matrix[:3, 3] = np.dot(matrix, np.array([0, 0, gripper_offset, 1]).T)[:3]

        matrices.append(matrix[:3].flatten())

    matrices = np.vstack(matrices)

    # Uncomment to view the generated grasp candidates
    plot_mesh_with_normals(mesh, matrices, up_vector)
    plt.show()

    return matrices
"""


def generate_candidates(mesh, grasps, num_samples=10, noise_level=0.05,
                        gripper_offset=-0.1, augment=True):

    # Defines the up-vector for the workspace frame
    up_vector = np.asarray([1, 0, 0])

    #points = [np.array([-0.01259965,  0.00669609, -0.00649286])]

    matrices = []
    print(len(grasps))
    for c1, c2 in grasps:
        #print('c1+c2')
        #print(c1)
        #print(c2)
        #c1 = np.array([-0.01259965,  0.00669609, -0.00649286])
        #c2 = np.array([ 0.00945797, -0.01483555, -0.00138991])
        #c1 = np.array([ 1, 0, 0])
        #c2 = np.array([ 0, 0, 1])
        normal = lib.utils.normalize_vector(c2-c1)

        # print(normal, mesh.triangles_cross[face])

        # Add random noise to the surface normals, centered around 0
        #if augment is True:
        #    normal += np.random.uniform(-noise_level, noise_level)
        #    normal = lib.utils.normalize_vector(normal)

        # Since we need to set a pose for the gripper, we need to calculate the
        # rotation matrix from a given surface normal
        
        matrix = lib.utils.get_rot_mat(up_vector, normal)
        #matrix = np.array([[0., -1., 0., 0.],
        #                   [1., 0., 0., 0.],
        #                   [0., 0., 1., 0.],
        #                   [0., 0., 0., 1.],])
        matrix[:3, 3] = (c1+c2)/2
        #theta_x = math.tan(matrix[2][1]/matrix[2][2])
        #theta_y = math.tan((-matrix[2][0])/(math.sqrt(math.pow(matrix[2][1],2) + math.pow(matrix[2][2],2))))
        #matrix_x = np.array([[1, 0,                  0                 ],
        #                     [0, math.cos(theta_x), -math.sin(theta_x) ],
        #                     [0, math.sin(theta_x),  math.cos(theta_x) ]])
        #matrix_y = np.array([[math.cos(theta_x),  0,  math.sin(theta_x)],
        #                     [0,                  1,  0                ],
        #                     [-math.sin(theta_x), 0,  math.cos(theta_x)]])
        #matrix = np.eye(4,4)
        #matrix[:3,:3] = np.matmul(matrix_y, matrix_x)
        #matrix[:3,3] = (c1+c2)/2


        # Calculate an offset for the gripper from the object.
        #matrix[:3, 3] = np.dot(matrix, np.array([0, 0, gripper_offset, 1]).T)[:3]
        #matrix[:3, 3] = np.dot(matrix, np.array([0, 0, -0.1, 1]).T)[:3]
        #matrix[2, 2] = matrix[2, 2] + 1

        matrices.append(matrix[:3].flatten())

    #print(len(matrices))
    matrices = np.vstack(matrices)
    
    #plot_mesh_with_normals(mesh, matrices, up_vector)
    #plot_mesh_with_points(mesh, grasps, up_vector)
    plt.show()
    
    return matrices


def run_grasps(sim, initial_pose, gripper_poses, object_name):

    #initial_pose = sim.get_object_pose()
    #initial_pose[:3, 3] = [0, 0, initial_height]
    #initial_pose = [[ 0.14091686,  0.77899223, -0.6109939 ,-2.58022399e-05],
    #            [ 0.98976118, -0.12500032,  0.06890373, 1.46757865e-03],
    #            [-0.02269896, -0.61444774, -0.78863092, 1.46522531e-02],
    #            [ 0         ,  0         ,  0         ,1]]
    # set height
    #initial_pose[2][3] = 0
    sim.run_threaded_drop(initial_pose)

    # Reset the object on each grasp attempt to its resting pose. Note this
    # doesn't have to be done, but it avoids instances where the object may
    # subsequently have fallen off the table
    object_pose = sim.get_object_pose()

    for count, row in enumerate(gripper_poses):
        print('candidate: ', count)

        work2candidate = gripper_poses[count]
        #work2candidate = np.dot(object_pose, work2candidate)

        sim.set_object_pose(object_pose[:3].flatten())

        # We can randomize the gripper candidate by rotation or translation.
        # Here we let the pose vary +- 3cm along local z, and a random
        # rotation between[0, 360) degress around local z
        #random_pose = lib.utils.randomize_pose(work2candidate,
        #                                       offset_mag=candidate_offset_mag,
        #                                       local_rot=candidate_local_rot)
        #sim.set_gripper_pose(random_pose)
        time.sleep(3)
        # set gripper to exact pose
        collision = sim.set_gripper_pose(work2candidate)
        if collision:
            print('grasp is colliding')
            time.sleep(2)
            continue
        # wait a bit before checking collisions and closing the gripper
        time.sleep(2)

        sim.set_camera_pose(work2candidate)
        rgb_image, depth_image = sim.camera_images()

        if rgb_image is None:
                raise Exception('No image returned.')

        print('sim.run_threaded_candidate()')
        grasp_res = sim.run_threaded_candidate()
        print('AFTER sim.run_threaded_candidate()')

        # SUCCESS
        if (grasp_res == '0'):
            postfix = 'success_%s_%d' % (object_name, count)
        else:
            postfix = 'fail_%s_%d' % (object_name, count)
        save_images(rgb_image, depth_image, postfix, '/home/silvia/dex-net/v-rep-grasping/output/images')

        




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


    #candidates = generate_candidates(mesh, num_samples=num_candidates,
    #                                 noise_level=candidate_noise_level,
    #                                 gripper_offset=candidate_offset)
    # mass = mesh.mass_properties['mass'] * 10
    # com = mesh.mass_properties['center_mass']
    # inertia = mesh.mass_properties['inertia'] * 5
    
    for i in range(5):
        print('OBJECT ', i)
        object_name = 'example' + str(i)
        mesh_path = '/home/silvia/dex-net/generated_shapes/' + object_name + '.obj'
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
        time.sleep(2)

        for pose_id in range(3):
            print('POSE ', i)
            # ----------- GET POSES AND GRASPS FOR GIVEN OBJECT AND POSE -----------
            initial_pose = db.get_stable_pose(object_name, pose_id)
            grasps, gripper_poses = db.stable_pose_grasps(object_name, pose_id)
            if grasps == None:
                continue
            # ----------------------------------------------------------------------

            run_grasps(sim, initial_pose, gripper_poses[:6], object_name)


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
