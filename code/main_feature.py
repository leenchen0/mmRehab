import os
import cv2
import time
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm

from util.movie_generation import movieMaker
from if_generation.generator import Generator, Method
from signal_processing.object_detection import threshold
from signal_processing.doppler_processing import fine_motion
from signal_processing.heatmap_generation import generate_rd, generate_rae, generate_ra
from signal_processing.angle_processing import AngleEstimator, AoA, Pointcloud_format
from signal_processing.radar_util import read_config, TI_RADAR_TYPE, range_freq_to_dis, doppler_freq_to_vel


import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_jobs', default=1, type=int)
parser.add_argument('--job_no', default=0, type=int)
args = parser.parse_args()

def cal_range(n, n_jobs, job_no):
    rfrom = n // n_jobs * job_no + min(job_no, n % n_jobs)
    rto = n // n_jobs * (job_no + 1) + min(job_no + 1, n % n_jobs)
    return rfrom, rto

"""
Param
"""
config_file             = './config_files/iwr1843boost.cfg'
layout                  = TI_RADAR_TYPE.iwr1843boost
parsing_debug           = True
clutter_removal_enabled = True
make_movie              = False
format                  = Pointcloud_format.cartesian
th                      = 0.30
is_synthesis            = False
fine_frame_len          = 64

"""
File Parsing: Radar config file
"""
radar_config    = read_config(config_file=config_file, layout=layout, debug=parsing_debug)

# Directory path
directory = '../../datasets/mmrehab'

# Traverse directory and subdirectories
adc_data_paths = []
data_paths     = []
for root, _, files in os.walk(directory):
    for file in files:
        # Check if file is adc_data.pkl
        if file.endswith('adc_data.pkl'):
            # Process the adc_data.pkl file
            data_paths.append(root.replace('/radar', ''))
            adc_data_paths.append(os.path.join(root, file))
print(data_paths)
print(f"path len: {len(data_paths)}")
# for i in range(9, 10):
rfrom, rto = cal_range(len(data_paths), args.num_jobs, args.job_no)
for i in range(rfrom, rto):

    data_path = data_paths[i]
    print(f"#{i+1}: {data_path}")
    """
    ADC Data:
        is_synthesis=True:  Pkl file
        is_synthesis=False: ADC raw data file
    """

    if is_synthesis:
        pkl_path  = data_path.replace('radar', 'images/results/demo_images.pkl')
        save_path = data_path.replace('syn_mmwave_rgb', 'fake_features').replace('radar', '')

        if not os.path.exists(pkl_path):
            continue
        duty_time = (radar_config['n_tx']*radar_config['n_loop'])*(radar_config['idle_time_sec']+radar_config['ramp_end_time_sec'])
        config = {
            'rx': radar_config['rx'],
            'samples_per_chirp': radar_config['n_sample'],
            'chirps_per_frame': radar_config['n_loop'],
            'start_freq': radar_config['start_freq_hz'],
            'adc_start_time': radar_config['adc_start_time_sec'],
            'ADC_rate': radar_config['ADC_rate'],
            'chirp_time': radar_config['n_sample']/radar_config['ADC_rate'],
            'slope': radar_config['freq_slope_hz_sec'],
            'fps': radar_config['n_loop'] / duty_time,
            'duty_time': duty_time,
            'n_frames': 1,
            'layout': layout,
            'noise': 0,
        }

        generator   = Generator(pkl_path=pkl_path, radar_config=config, method=Method.mesh, radar_pos=(0, -2, 0))
        adc_frames  = generator.run()                               # (n_frame, n_virtual_ant, n_loop, n_sample)
        adc_frames  = np.transpose(adc_frames, axes=(0, 2, 1, 3))   # (n_frame, n_loop, n_virtual_ant, n_sample)
        print(f"adc_frames shape: {np.array(adc_frames).shape}")

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'adc_frames.pkl'), 'wb') as f:
            pickle.dump(adc_frames, f)
    else:
        rgb_path            = os.path.join(data_path, 'images/video.mp4')
        adc_data_path       = adc_data_paths[i]
        save_path           = data_path.replace('syn_mmwave_rgb', 'real_features')
        with open(adc_data_path, 'rb') as f:
            adc_frames = pickle.load(f)

    n_range_fft     = 256 # adc_frames[0].shape[2]
    n_doppler_fft   = 256 # adc_frames[0].shape[0]
    dis = range_freq_to_dis(radar_config['freq_slope_hz_sec'], radar_config['n_sample'], radar_config['ADC_rate'], n_range_fft)
    vel = doppler_freq_to_vel(radar_config['n_tx'], radar_config['n_loop'], radar_config['idle_time_sec'],
                              radar_config['ramp_end_time_sec'], radar_config['start_freq_hz'], n_doppler_fft)
    n_frame = adc_frames.shape[0]

    # min_range_id    = n_range_fft//2
    # max_range_id    = np.argmax(dis)
    min_range_id    = np.argmin(dis)
    max_range_id    = n_range_fft//2 - 1
    dis             = dis[min_range_id: max_range_id + 1]
    r_range_id      = {'min': min_range_id, 'max': max_range_id}

    """
    Dynamic Scene
    # 1. Range-Azimuths
    # 2. Peak Detection
    # 3. Elevation
    # 4. Doppler
    # 5. Range-Elevation
    # 6. Range-Doppler
    Static Scene
    # 1. Range-Azimuth-Elevation
    # 2. Peak Detection
    """

    radar_cubes             = []    # (n_frame, n_loop, n_virtual_ant, n_range_bin)
    peak_frames             = []
    range_azimuths          = []    # (n_frame, n_range_bin, n_aoa_fft)
    range_elevations        = []    # (n_frame, n_range_bin, n_aoa_fft)
    range_dopplers          = []    # (n_frame, n_range_bin, n_doppler_bin)
    static_features         = []    # (n_frame, n_range_bin, n_aoa_fft, n_aoa_fft)
    pointclouds             = []
    dynamic_pcs             = []
    static_pcs              = []
    static_clutters         = np.empty((n_frame, adc_frames.shape[2], dis.shape[0]), dtype=complex)    # (n_frame, n_virtual_ant, n_range_bin)
    fine_dopplers           = np.zeros((n_frame, dis.shape[0], fine_frame_len))

    for f_i in range(n_frame):
        s = time.perf_counter()

        adc_frame = adc_frames[f_i]
        # 1. Range-Azimuths
        # s = time.perf_counter()
        range_azimuth, radar_cube, static_clutter = generate_ra(adc_frame, r_range_id=r_range_id, n_range_fft=n_range_fft, radar_config=radar_config, clutter_removal_enabled=clutter_removal_enabled, method=AoA.mvdr)
        # print(f"[Range-Azimuths] {time.perf_counter()-s}")
        # 2. Peak Detection
        # s = time.perf_counter()
        peaks = threshold(range_azimuth, th=th)
        # print(f"[Peak Detection] {time.perf_counter()-s}")
        # 3. Elevation
        # s = time.perf_counter()
        if radar_cube.shape[0] != radar_config['rx'].shape[0]:
            radar_cube = np.transpose(radar_cube, (1, 0, 2))    # [n_rx, n_chirp, n_range_fft]
        angle_estimator = AngleEstimator(radar_config, r_range_id=r_range_id, n_range_fft=n_range_fft, n_doppler_fft=n_doppler_fft)
        rae_index      = angle_estimator.bf_per_point_elevation(radar_cube, peaks, method=AoA.mvdr, format=Pointcloud_format.index)
        # print(f"[Elevation] {time.perf_counter()-s}")
        # 4. Doppler
        # s = time.perf_counter()
        dynamic_pc = angle_estimator.bf_per_point_doppler(radar_cube, rae_index, format=format)
        # print(f"[Doppler] {time.perf_counter()-s}")
        # 5. Range-Elevation
        # s = time.perf_counter()
        __, range_elevation, __, __ = generate_rae(adc_frame, r_range_id=r_range_id, n_range_fft=n_range_fft, radar_config=radar_config, clutter_removal_enabled=clutter_removal_enabled, method=AoA.mvdr)
        # print(f"[Range-Elevation] {time.perf_counter()-s}")
        # 6. Range-Doppler
        # s = time.perf_counter()
        det_matrix, __, __, __ = generate_rd(adc_frame, r_range_id=r_range_id, n_range_fft=n_range_fft, n_doppler_fft=n_doppler_fft, clutter_removal_enabled=clutter_removal_enabled)
        # print(f"[Range-Doppler] {time.perf_counter()-s}")

        # 1. Range-Azimuth-Elevation & 2. Peak Detection
        # s = time.perf_counter()
        static_rae, static_pc = angle_estimator.bf_per_range_static(static_clutter, method=AoA.conventional, format=format)
        # print(f"[Static Scene] {time.perf_counter()-s}")

        # fine-motion
        if f_i >= fine_frame_len:
            fine_doppler = fine_motion(static_clutters[f_i-fine_frame_len:f_i])
            fine_dopplers[f_i] = fine_doppler.T
        print(f"[Signal processing of Frame #{f_i}] {time.perf_counter()-s}")
        print(f'range_azimuth: {range_azimuth.shape}, range_elevation: {range_elevation.shape}, range_doppler: {det_matrix.shape}, static_clutter: {static_clutter.shape}, fine_doppler: {fine_dopplers[f_i].shape}')

        static_clutters[f_i] = static_clutter
        pc = np.concatenate((dynamic_pc, static_pc), axis=0)
        radar_cubes.append(radar_cube)
        peak_frames.append(peaks)
        static_features.append(static_rae)
        range_azimuths.append(range_azimuth)
        range_elevations.append(range_elevation)
        range_dopplers.append(det_matrix)
        pointclouds.append(pc)
        dynamic_pcs.append(dynamic_pc)
        static_pcs.append(static_pc)
        print(f"static/dynamic: {static_pc.shape[0]}/{dynamic_pc.shape[0]}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'radar_cubes.pkl'), 'wb') as f:
        pickle.dump(radar_cubes, f)
    with open(os.path.join(save_path, 'peak_frames.pkl'), 'wb') as f:
        pickle.dump(peak_frames, f)
    with open(os.path.join(save_path, 'static_clutters.pkl'), 'wb') as f:
        pickle.dump(static_clutters, f)
    with open(os.path.join(save_path, 'static_features.pkl'), 'wb') as f:
        pickle.dump(static_features, f)
    with open(os.path.join(save_path, 'range_azimuths.pkl'), 'wb') as f:
        pickle.dump(range_azimuths, f)
    with open(os.path.join(save_path, 'range_elevations.pkl'), 'wb') as f:
        pickle.dump(range_elevations, f)
    with open(os.path.join(save_path, 'range_dopplers.pkl'), 'wb') as f:
        pickle.dump(range_dopplers, f)
    with open(os.path.join(save_path, 'dynamic_pcs.pkl'), 'wb') as f:
        pickle.dump(dynamic_pcs, f)
    with open(os.path.join(save_path, 'static_pcs.pkl'), 'wb') as f:
        pickle.dump(static_pcs, f)
    with open(os.path.join(save_path, 'fine_dopplers.pkl'), 'wb') as f:
        pickle.dump(fine_dopplers, f)

    if make_movie:
        s = time.perf_counter()
        motion_Hz   = np.arange(-fine_frame_len/2, fine_frame_len/2) / fine_frame_len * radar_config['fps_s']
        n_aoa_fft   = 64
        fft_freq_a  = np.arange(-1, 1, 2/n_aoa_fft)
        angle_phi   = np.arcsin(fft_freq_a)
        ims = []
        lim = [-3, 3]
        fig = plt.figure(figsize=(25, 10))

        ax_ra   = fig.add_subplot(251)
        ax_od   = fig.add_subplot(252)
        ax_rd   = fig.add_subplot(253)
        ax_re   = fig.add_subplot(254)
        ax_sc   = fig.add_subplot(255)
        ax_pc   = fig.add_subplot(256, projection='3d')
        if is_synthesis:
            ax_rgb  = fig.add_subplot(257, projection='3d')
        else:
            cap = cv2.VideoCapture(rgb_path)
            ax_rgb  = fig.add_subplot(257)
        ax_sra  = fig.add_subplot(258)
        ax_sre  = fig.add_subplot(259)
        ax_fm   = fig.add_subplot(2, 5, 10)

        ax_ra.set_xlabel('Azimuth (deg)')
        ax_ra.set_ylabel('Range (m)')
        ax_ra.set_title(f'Range-Azimuth Heatmap')

        ax_sra.set_xlabel('Azimuth (deg)')
        ax_sra.set_ylabel('Range (m)')
        ax_sra.set_title(f'[Static] Range-Azimuth Heatmap')

        ax_od.set_xlabel('Azimuth (deg)')
        ax_od.set_ylabel('Range (m)')
        ax_od.set_title(f'Range-Azimuth Heatmap')

        ax_rd.set_xlabel('Range (m)')
        ax_rd.set_ylabel('Doppler velocity (m/s)')
        ax_rd.set_title(f'Range-Doppler Heatmap')

        ax_re.set_xlabel('Elevation (deg)')
        ax_re.set_ylabel('Range (m)')
        ax_re.set_title(f'Range-Elevation Heatmap')

        ax_sre.set_xlabel('Elevation (deg)')
        ax_sre.set_ylabel('Range (m)')
        ax_sre.set_title(f'[Static] Range-Elevation Heatmap')

        ax_sc.set_xlabel('Range (m)')
        ax_sc.set_ylabel('Antenna #')
        ax_sc.set_title(f'Static Clutter')

        ax_fm.set_xlabel('Frequency (Hz)')
        ax_fm.set_ylabel('Range (m)')
        ax_fm.set_title(f'Fine Doppler')

        ax_pc.set_xlabel('X (m)')
        ax_pc.set_ylabel('Y (m)')
        ax_pc.set_zlabel('Z (m)')
        ax_pc.set_xlim(lim)
        ax_pc.set_ylim(lim)
        ax_pc.set_zlim(lim)

        if is_synthesis:
            ax_rgb.set_xlabel('X (m)')
            ax_rgb.set_ylabel('Y (m)')
            ax_rgb.set_zlabel('Z (m)')
            ax_rgb.set_xlim(lim)
            ax_rgb.set_ylim(lim)
            ax_rgb.set_zlim(lim)

        for f_i in tqdm(range(n_frame), desc="movie making..."):
            static_clutter = static_clutters[f_i]
            fine_doppler   = fine_dopplers[f_i]
            if static_clutter is None:
                static_clutter = np.zeros((radar_config['rx'].shape[0], n_range_fft))
            static_clutter  = 20*np.log10(np.abs(static_clutter))
            static_ra       = np.sum(static_features[f_i], axis=1)
            static_re       = np.sum(static_features[f_i], axis=2)
            range_azimuth   = range_azimuths[f_i]
            range_elevation = range_elevations[f_i]
            range_doppler   = range_dopplers[f_i]
            pointcloud      = pointclouds[f_i]
            peaks           = peak_frames[f_i]


            det_mask    = np.zeros(range_azimuth.shape)
            for d, r, _ in peaks.astype(int):
                det_mask[d, r] = 1

            if is_synthesis:
                vertices, __    = generator.get_mesh_data(f_i)
                point_array, __ = generator.select_vertices(vertices)
                ims.append((
                    ax_ra.pcolormesh(angle_phi/np.pi*180, dis, range_azimuth),
                    ax_od.pcolormesh(angle_phi/np.pi*180, dis, det_mask),
                    ax_re.pcolormesh(angle_phi/np.pi*180, dis, range_elevation),
                    ax_rd.pcolormesh(dis, vel, range_doppler.T),
                    ax_sc.pcolormesh(dis, [i+1 for i in range(static_clutter.shape[0])], static_clutter),
                    ax_sra.pcolormesh(angle_phi/np.pi*180, dis, static_ra),
                    ax_sre.pcolormesh(angle_phi/np.pi*180, dis, static_re),
                    ax_pc.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=pointcloud[:, 3]),
                    ax_rgb.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='r'),
                    ax_rgb.scatter(point_array[:, 0], point_array[:, 1], point_array[:, 2], c='b'),
                    ax_fm.pcolormesh(motion_Hz, dis, fine_doppler)
                    ))
            else:
                ret, img = cap.read()
                ims.append((
                    ax_ra.pcolormesh(angle_phi/np.pi*180, dis, range_azimuth),
                    ax_od.pcolormesh(angle_phi/np.pi*180, dis, det_mask),
                    ax_re.pcolormesh(angle_phi/np.pi*180, dis, range_elevation),
                    ax_rd.pcolormesh(dis, vel, range_doppler.T),
                    ax_sc.pcolormesh(dis, [i+1 for i in range(static_clutter.shape[0])], static_clutter),
                    ax_sra.pcolormesh(angle_phi/np.pi*180, dis, static_ra),
                    ax_sre.pcolormesh(angle_phi/np.pi*180, dis, static_re),
                    ax_pc.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], c=pointcloud[:, 3]),
                    ax_rgb.imshow(img),
                    ax_fm.pcolormesh(motion_Hz, dis, fine_doppler)
                    ))

        movieMaker(fig, ims,  os.path.join(save_path, 'display.mp4'))
        print(f"[Make Movie] {time.perf_counter()-s}")
