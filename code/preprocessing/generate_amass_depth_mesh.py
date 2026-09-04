import sys
sys.path.append('..')
import argparse
import time

import os
import re
from os.path import join as fullfile

import pickle

import numpy as np
from matplotlib import pyplot as plt
import tqdm
from scipy.spatial import ConvexHull
import torch
from tqdm import tqdm
from numba import jit
import threading
import multiprocessing
from multiprocessing import Process
from scipy.spatial.transform import Rotation as R

from util.utils import rotation6d_2_rot_mat, rodrigues_2_rot_mat
from smpl.smpl_utils_extend import SMPL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='')

# Exp config
parser.add_argument('--dir', default='../../../AMASS/raw', type=str)
parser.add_argument('--save_dir', default='../../datasets/amass', type=str)
parser.add_argument('--setting', default='depth_mesh', type=str)
parser.add_argument('--max_threads', default=1, type=int)
parser.add_argument('--img_dim', default=96, type=int)
parser.add_argument('--no_trans', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--video', action='store_true')

args = parser.parse_args()

radar_pos = (0, -2, 0)


def make_folder(filename):
    folder_path = os.path.dirname(filename)
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except Exception as e:
            print(f"An exception occurred: {e}")


def movieMaker(fig, ims, save_dir):
    import matplotlib.animation as animation
    # Set up formatting for the Range Azimuth heatmap movies
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    print('Complete')


def HPR(p, C, param=2):
    # HPR - Using HPR ("Hidden Point Removal") method, approximates a visible subset of points
    # as viewed from a given viewpoint.
    #
    # Usage: visiblePtInds = HPR(p, C, param)
    #
    # Input:
    # p - NxD D dimensional point cloud.
    # C - 1xD D dimensional viewpoint.
    # param - parameter for the algorithm. Indirectly sets the radius.
    #
    # Output:
    # visiblePtInds - indices of p that are visible from C.

    dim = p.shape[1]
    numPts = p.shape[0]

    # Move C to the origin
    p = p - np.tile(C, (numPts, 1))

    # Calculate ||p||
    normp = np.sqrt(np.sum(p**2, axis=1))[:, np.newaxis]

    # Sphere radius
    R = np.tile(np.max(normp) * (10**param), (numPts, 1))

    # Spherical flipping
    P = p + 2 * np.tile(R - normp, (1, dim)) * p / np.tile(normp, (1, dim))

    # Convex hull
    hull_points = np.vstack([P, np.zeros((1, dim))])
    hull = ConvexHull(hull_points)
    visiblePtInds = np.unique(hull.vertices)

    # Remove the artificial point added to the convex hull
    visiblePtInds = visiblePtInds[visiblePtInds != numPts]

    return visiblePtInds


def select_vertices(vertices):
    v_indices   = HPR(vertices, np.array(radar_pos))
    v_array     = vertices[v_indices, :]

    return v_array, v_indices


def gen_scene(vertices):
    point_array, vert_indices = select_vertices(vertices)

    return point_array


def gen_depth_mesh(frame):
    point_array = gen_scene(frame['mesh'])
    img_dim = args.img_dim
    depth_mesh = np.zeros((img_dim, img_dim))

    scale = 50 / abs(radar_pos[1])
    for point in point_array:
        dis = point[1]

        xi = min(img_dim - 1, max(0, int(round(point[0] * scale)) + img_dim // 2))
        yi = min(img_dim - 1, max(0, int(round(point[2] * scale)) + img_dim // 2))

        depth_mesh[yi, xi]  = dis

    return depth_mesh, point_array


def gen_vertices(smpl, pose, beta):
    r = R.from_rotvec(pose[:3])
    r2 = R.from_euler('x', 90, degrees=True)
    r3 = r2 * r
    pose[:3] = r3.as_rotvec()

    pose_param = torch.from_numpy(np.concatenate((pose[:(21*3)], np.zeros((3*3, ))), dtype=np.float32))
    pose_param = pose_param.unsqueeze(0)
    pose_param = rodrigues_2_rot_mat(pose_param.to(device))
    pose_param = pose_param.reshape(1, -1, 3, 3)

    beta = torch.tensor(beta).unsqueeze(0)
    batch_vertices, batch_joints = smpl(beta.float(), pose_param.float(), torch.zeros((1, 3)).float())
    pred = batch_vertices[0].detach().cpu().numpy()
    pred = pred[:, [0, 2, 1]]
    pred[:, 2] = -pred[:, 2]
    return pred


def process_one_dir(folder, save_folder):
    smpl = SMPL('../smpl/smpl_m.pkl').to(device)

    files = os.listdir(folder)
    poses = []
    for f in files:
        if f.endswith('npz'):
            poses.append(f)
    print(poses)

    for pose in poses:
        if not os.path.exists(fullfile(save_folder, pose[:-4])):
            continue

        if args.video or args.test:
            fig = plt.figure(figsize=(15, 5))
            ax_points = fig.add_subplot(121, projection='3d')
            ax_depth_mesh = fig.add_subplot(122)
            imgs = []

        all_frames = np.load(fullfile(folder, pose))
        beta = all_frames['betas']
        poses_param = all_frames['poses']

        save_dir = fullfile(save_folder, pose[:-4], args.setting)

        num_frames = poses_param.shape[0]
        label_downsample = 6
        for frame_no in tqdm(range(0, num_frames, label_downsample), 'Progress'):
            frame = frame_no
            save_path = fullfile(save_dir, f'frame_{frame // label_downsample + 2}.npy')

            if (not args.overwrite) and (os.path.exists(save_path)):
                continue

            try:
                depth_mesh, points = gen_depth_mesh({
                    'mesh': gen_vertices(smpl, poses_param[frame], np.zeros((10,))),
                })
                # Save the current frame
                make_folder(save_path)
                np.save(save_path, depth_mesh)
            except Exception as e:
                print(e)

            if args.video:
                imgs.append((ax_points.scatter(points[:, 0], points[:, 1], points[:, 2]), ax_depth_mesh.pcolormesh(depth_mesh)))
            if args.test:
                ax_points.scatter(points[:, 0], points[:, 1], points[:, 2])
                ax_depth_mesh.pcolormesh(depth_mesh)
                plt.show()
                break

        if args.video:
            movieMaker(fig, imgs, fullfile(save_folder, f'{pose[:-4]}_depth_mesh.mp4'))
        if args.test:
            break


def all_subdirs(path):
    files = os.listdir(path)
    dirs = []
    for f in files:
        if f.startswith('.') or (not os.path.isdir(fullfile(path, f))):
            continue

        dirs.append(f)
    return dirs


def main():
    folder = args.dir

    sub_datasets = all_subdirs(folder)
    print(sub_datasets)

    threads = []
    for i, dataset in enumerate(sub_datasets):
        print(f'[{i + 1}/{len(sub_datasets)}] Processing {dataset}...')

        people = all_subdirs(fullfile(folder, dataset))
        print(people)
        for person in people:
            save_dir = fullfile(args.save_dir, dataset, person)

            while len(threads) >= args.max_threads:
                time.sleep(0.5)
                threads = list(filter(lambda t: t.is_alive(), threads))

            thread = Process(target=process_one_dir, args=[fullfile(folder, dataset, person), save_dir])
            thread.start()
            threads.append(thread)

            if args.test:
                break
        if args.test:
            break

    for t in threads:
        t.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()



