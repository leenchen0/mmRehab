import sys
import time
import torch
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from enum import Enum
from if_generation.point import Point
from if_generation.motionbase import Line
from if_generation.simulator import Simulator
from if_generation.util import HPR, moving_average
from smpl.smpl_utils_extend import SMPL
from if_generation.radar_util import Radar, cal_center_freq
from human_body_prior import *
from human_body_prior.tools.omni_tools import copy2cpu as c2c

DEVICE  = ("cuda" if torch.cuda.is_available() else "cpu")
MAF_SIZE    = 5
Mode    = Enum('Mode', 'pointcloud heatmap')
Method  = Enum('Method','mesh skeleton test')
class Generator:
    def __init__(self, pkl_path, radar_config, save_folder=None, method = Method.mesh, point_array=None, vel_array=None, radar_pos=(0, -1, 0), mesh_fps=10, debug=False, is_Hupr=True):
        self.save_folder    = save_folder
        self.method         = method
        self.mesh_fps       = mesh_fps
        self.radar_pos      = radar_pos
        self.radar_config   = radar_config
        self.debug          = debug
        self.is_Hupr        = is_Hupr

        if not self.method == Method.test:
            if self.is_Hupr:
                self.load_smpl()
                self.frames     = joblib.load(pkl_path)
                self.frame_len  = len(self.frames)
            else:
                with open(pkl_path, 'rb') as f:
                    data            = pickle.load(f) # {'mesh': verts, 'mocap_framerate': mocap_framerate}
                    mocap_fps       = data['mocap_framerate']
                    self.frames     = data['mesh'][::int(mocap_fps//self.mesh_fps)]
                    y_min           = np.min(self.frames[:, :, 1])
                    y_max           = np.max(self.frames[:, :, 1])
                    self.radar_pos  = (0, y_min-2, 0)
                    print(f"y_min: {y_min} y_max: {y_max} radar_pos: {self.radar_pos}")
                    self.frame_len  = self.frames.shape[0]
            if self.method == Method.mesh:
                self.velocity_meshs              = None
                self.smoothed_velocity_meshs     = None
            elif self.method == Method.skeleton:
                self.velocity_skeletons          = None
                self.smoothed_velocity_skeletons = None
            self.cal_velocity_all_frames()
        else:
            self.point_array            = point_array
            self.vel_array              = vel_array
            self.frame_len              = self.radar_config['n_frames']
            self.radar_config['noise']  = None

        self.init_radar()
    def run(self, frame_range=None):
        if frame_range is None:
            frame_range = range(self.frame_len)

        signals = np.zeros((self.frame_len, self.radar_config['rx'].shape[0],
                            self.radar_config['chirps_per_frame'], self.radar_config['samples_per_chirp']), dtype=complex)
        for f_i in tqdm(frame_range, desc="Generate mmwave among all frame..."):
            scene, __, __   = self.gen_scene(f_i)
            if_signal       = self.gen_if_signal(scene)
            signals[f_i] = if_signal

        return signals

    def init_radar(self):
        adc_start_time  = self.radar_config['adc_start_time']
        num_sample      = self.radar_config['samples_per_chirp']
        ADC_rate        = self.radar_config['ADC_rate']
        start_freq      = self.radar_config['start_freq']
        chirp_time      = self.radar_config['chirp_time']
        slope           = self.radar_config['slope']
        noise           = self.radar_config['noise']
        center_freq     = cal_center_freq(start_freq, adc_start_time, slope, num_sample, ADC_rate)

        space = 3e8/center_freq/2    # wavelength / 2
        self.radars = []
        rx = self.radar_config['rx']*space
        for pos in rx:
            self.radars.append(Radar(Tx_pos=self.radar_pos, Rx_pos=self.radar_pos+pos, f1=start_freq,
                                     slope=slope, ADC_rate=ADC_rate, chirp_time=chirp_time, noise=noise))

    def cal_velocity_all_frames(self):
        if self.method == Method.mesh:
            self.velocity_meshs = []
        elif self.method == Method.skeleton:
            self.velocity_skeletons = []

        for f_i in tqdm(range(1, self.frame_len), desc="Calculate velocity among all frame..."):
            mesh_pre, skeleton_pre  = self.get_mesh_data(f_i-1)
            mesh_cur, skeleton_cur  = self.get_mesh_data(f_i)

            if self.method == Method.mesh:
                velocity_mesh = self.cal_velocity_single_frames(mesh_pre, mesh_cur)
                if not self.velocity_meshs:
                    self.velocity_meshs.append(np.zeros_like(velocity_mesh))
                self.velocity_meshs.append(velocity_mesh)
            elif self.method == Method.skeleton:
                velocity_skeleton = self.cal_velocity_single_frames(skeleton_pre, skeleton_cur)
                if not self.velocity_skeletons:
                    self.velocity_skeletons.append(np.zeros_like(velocity_skeleton))
                self.velocity_skeletons.append(velocity_skeleton)

        if self.method == Method.mesh:
            self.velocity_meshs                 = np.array(self.velocity_meshs)
            self.smoothed_velocity_meshs        = moving_average(self.velocity_meshs, window_size=MAF_SIZE)
        elif self.method == Method.skeleton:
            self.velocity_skeletons             = np.array(self.velocity_skeletons)
            self.smoothed_velocity_skeletons    = moving_average(self.velocity_skeletons, window_size=MAF_SIZE)

    def cal_velocity_single_frames(self, frame_pre, frame_cur):



        if len(frame_pre) != len(frame_cur):
            raise ValueError("Meshes must have the same number of vertices")

        time_interval = 1/self.mesh_fps

        velocities = []
        for vertex_pre, vertex_cur in zip(frame_pre, frame_cur):
            dis         = vertex_cur - vertex_pre
            velocity    = dis / time_interval
            velocities.append(velocity)

        return np.array(velocities)

    def gen_scene(self, i):
        if self.method == Method.mesh:
            vertices, __ = self.get_mesh_data(i)
            point_array, vert_indices = self.select_vertices(vertices)
            vel_array = self.smoothed_velocity_meshs[i, vert_indices, :]
        elif self.method == Method.skeleton:
            __, skeleton = self.get_mesh_data(i)
            point_array = skeleton
            vel_array   = self.smoothed_velocity_skeletons[i, :, :]
        elif self.method == Method.test:
            point_array = self.point_array
            vel_array   = self.vel_array

        # print(f"[Generator-gen_scene] vert_array.shape: {vert_array.shape} vel_array.shape: {vel_array.shape}")
        if not self.method == Method.test and i == 0:
            scene = [Point(point_array)]
        else:
            scene = []
            for (vert, vel) in zip(point_array, vel_array):
                scene.append(Point(vert[np.newaxis, :], motion=Line(vel))) # Note: one point corresponds one motion
        return scene, point_array, vel_array

    def gen_if_signal(self, scene):
        duty_time           = self.radar_config['duty_time']    # Duty time in a frame
        fps                 = self.radar_config['fps']
        n_frames            = self.radar_config['n_frames']
        chirps_per_frame    = self.radar_config['chirps_per_frame']
        samples_per_chirp   = self.radar_config['samples_per_chirp']
        n_rx                = self.get_num_vitural_rx()

        # put them into the simulator
        simulator   = Simulator(self.radars, scene, duty_time, fps)
        if_signal   = simulator.run()
        if_signal   = if_signal.reshape((n_rx, n_frames, chirps_per_frame, samples_per_chirp))
        if_signal   = if_signal[:, 0]   # take the first frame

        return if_signal

    def select_vertices(self, vertices):
        v_indices   = HPR(vertices, np.array(self.radar_pos))
        v_array     = vertices[v_indices, :]

        return v_array, v_indices

    def get_mesh_data(self, i):
        if not self.is_Hupr:
            return self.frames[i], None

        frame           = self.get_frame(i)
        smpl_info       = frame['smpl']
        smpl1           = smpl_info[0]
        pos             = frame['camera'][0].copy()
        pos[2]          = pos[2]/40
        pos[[1, 2]]     = pos[[2, 1]]

        # Get the vertices of the mesh
        betas           = torch.from_numpy(np.expand_dims(smpl1['betas'], 0))
        global_orient   = np.dot(smpl1['global_orient'], np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32))
        pose            = torch.from_numpy(np.expand_dims(np.concatenate((global_orient, smpl1['body_pose']), axis=0), 0))
        batch_vertices, batch_skeleton = self.smpl(betas, pose, torch.zeros((1, 3)))
        vertices = batch_vertices[0].detach().cpu().numpy()
        skeleton = batch_skeleton[0].detach().cpu().numpy()

        # vertices[:, [1, 2]] =  vertices[:, [2, 1]]
        # vertices[:, 2]      = -vertices[:, 2]

        # skeleton[:, [1, 2]] =  skeleton[:, [2, 1]]
        # skeleton[:, 2]      = -skeleton[:, 2]

        vertices            = vertices + pos*0.5
        skeleton            = skeleton + pos*0.5

        return vertices, skeleton

    def load_smpl(self):
        # Load SMPL model
        self.smpl = SMPL('./smpl/smpl_m.pkl').to(DEVICE)
        self.faces = self.smpl.faces

    def get_frame(self, i):
        keys = list(self.frames.keys())
        return self.frames[keys[i]]

    def get_num_vitural_rx(self):
        return self.radar_config['rx'].shape[0]

    def plot_debug(self):
        fig = plt.figure(figsize=(15, 15))
        ax_mesh     = fig.add_subplot(221, projection='3d')
        ax_skeleton = fig.add_subplot(222, projection='3d')
        ax_vm       = fig.add_subplot(223)
        ax_vs       = fig.add_subplot(224)

        for f_i in tqdm(range(1, self.frame_len), desc="Plot debug..."):
            v, s = self.get_mesh_data(f_i)
            # Plot the 3D mesh
            ax_mesh.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=self.faces, cmap='viridis', edgecolor='k')
            ax_mesh.set_xlim(-1, 1)
            ax_mesh.set_ylim(-1, 1)
            ax_mesh.set_zlim(-1, 1)
            ax_mesh.set_xlabel('X')
            ax_mesh.set_ylabel('Y')
            ax_mesh.set_zlabel('Z')
            # Plot the 3D skeleton
            ax_skeleton.scatter(s[:, 0], s[:, 1], s[:, 2])
            ax_skeleton.set_xlim(-1, 1)
            ax_skeleton.set_ylim(-1, 1)
            ax_skeleton.set_zlim(-1, 1)
            ax_skeleton.set_xlabel('X')
            ax_skeleton.set_ylabel('Y')
            ax_skeleton.set_zlabel('Z')
            # Plot the velocity of mesh
            # ax_vm.plot(np.array([i for i in range(f_i)]), self.velocity_meshs[:f_i, 0], 'r-', label='X w/o filter')
            # ax_vm.plot(np.array([i for i in range(f_i)]), self.velocity_meshs[:f_i, 1], 'g-', label='Y w/o filter')
            # ax_vm.plot(np.array([i for i in range(f_i)]), self.velocity_meshs[:f_i, 2], 'b-', label='Z w/o filter')
            ax_vm.plot(np.array([i for i in range(f_i)]), self.smoothed_velocity_meshs[:f_i, 0, 0], 'r--', label='X w/ filter')
            ax_vm.plot(np.array([i for i in range(f_i)]), self.smoothed_velocity_meshs[:f_i, 0, 1], 'g--', label='Y w/ filter')
            ax_vm.plot(np.array([i for i in range(f_i)]), self.smoothed_velocity_meshs[:f_i, 0, 2], 'b--', label='Z w/ filter')
            ax_vm.set_xlim(0, self.frame_len)
            ax_vm.set_ylim(-0.5, 0.5)
            ax_vm.set_xlabel('Frame #')
            ax_vm.set_ylabel('Velocity (m/s)')
            # Plot the velocity of skeleton
            # ax_vs.plot(np.array([i for i in range(f_i)]), self.velocity_skeletons[:f_i, 0], 'r-', label='X w/o filter')
            # ax_vs.plot(np.array([i for i in range(f_i)]), self.velocity_skeletons[:f_i, 1], 'g-', label='Y w/o filter')
            # ax_vs.plot(np.array([i for i in range(f_i)]), self.velocity_skeletons[:f_i, 2], 'b-', label='Z w/o filter')
            ax_vs.plot(np.array([i for i in range(f_i)]), self.smoothed_velocity_skeletons[:f_i, 0, 0], 'r--', label='X w/ filter')
            ax_vs.plot(np.array([i for i in range(f_i)]), self.smoothed_velocity_skeletons[:f_i, 0, 1], 'g--', label='Y w/ filter')
            ax_vs.plot(np.array([i for i in range(f_i)]), self.smoothed_velocity_skeletons[:f_i, 0, 2], 'b--', label='Z w/ filter')
            ax_vs.set_xlim(0, self.frame_len)
            ax_vs.set_ylim(-0.5, 0.5)
            ax_vm.set_xlabel('Frame #')
            ax_vm.set_ylabel('Velocity (m/s)')

            plt.title(f'Frame #{f_i}')
            plt.pause(0.01)
            ax_mesh.clear()
            ax_skeleton.clear()
            ax_vm.clear()
            ax_vs.clear()

        plt.show()
