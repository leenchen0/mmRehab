import warnings
import numpy as np
import matplotlib.pyplot as plt
from signal_processing.dsp_util import Window
from signal_processing.range_processing import range_fft
from signal_processing.angle_processing import AngleEstimator, AoA
from signal_processing.doppler_processing import doppler_fft, clutter_removal

def generate_rd(adc_frame, r_range_id=None, n_range_fft=None, n_doppler_fft=None, clutter_removal_enabled=True, dis=None, vel=None, ax=None, is_plot=False):
    if n_range_fft is None:
        n_range_fft = adc_frame.shape[2]
    if n_doppler_fft is None:
        n_doppler_fft = adc_frame.shape[0]

    # Range FFT
    radar_cube = range_fft(adc_frame, n_range_fft=n_range_fft, window_type_1d=Window.HANNING)    # (n_loop, n_virtual_ant, n_range_bin)

    if r_range_id is None:
        r_range_id = {}
        r_range_id['min'] = 0
        r_range_id['max'] = n_range_fft-1

    radar_cube = radar_cube[:, :, r_range_id['min']: r_range_id['max']+1]
    # (Optional) Static Clutter Removal
    static_clutter = None
    if clutter_removal_enabled:
        radar_cube, static_clutter = clutter_removal(radar_cube, axis=0)
    # Doppler FFT
    det_matrix, rd_profile = doppler_fft(radar_cube, n_doppler_fft=n_doppler_fft, window_type_2d=Window.HANNING)   # (n_range_bin, n_doppler_bin)
    if is_plot:
        if dis is None or vel is None:
            print(f"dis and vel is None")
            return det_matrix, rd_profile, radar_cube, static_clutter
        plot_rd(dis, vel, det_matrix, ax['rd'])
        plot_sc(dis, static_clutter, ax['sc'])
        plt.pause(0.005)
    return det_matrix, rd_profile, radar_cube, static_clutter

def generate_rae(adc_frame, r_range_id=None, n_range_fft=None, radar_config=None, clutter_removal_enabled=True, method=AoA.mvdr, projection='raw', format='phi', ax=None, is_plot=False):
    if n_range_fft is None:
        n_range_fft = adc_frame.shape[2]
        
    radar_cube = range_fft(adc_frame, window_type_1d=Window.HANNING)    # (n_loop, n_virtual_ant, n_range_bin)

    if r_range_id is None:
        r_range_id = {}
        r_range_id['min'] = 0
        r_range_id['max'] = n_range_fft-1
    
    radar_cube = radar_cube[:, :, r_range_id['min']: r_range_id['max']+1]
    angle_estimator = AngleEstimator(radar_config, n_range_fft=n_range_fft, r_range_id=r_range_id)
    # (Optional) Static Clutter Removal
    static_clutter = None
    if clutter_removal_enabled:
        radar_cube, static_clutter = clutter_removal(radar_cube, axis=0)
    # per bin
        
    radar_cube = np.transpose(radar_cube, (1, 0, 2)) # [n_rx, n_chirp, n_range_fft]
    range_azimuth_elevation = angle_estimator.bf_per_range_2d(radar_cube, method=method)

    range_azimuth   = np.sum(range_azimuth_elevation, axis=1)
    range_elevation = np.sum(range_azimuth_elevation, axis=2)

    if is_plot:
        angle_estimator.plot_range_azimuth_heatmap(range_elevation, projection=projection, format=format, ax=ax['ra'])
        # plot_sc(angle_estimator.dis, static_clutter, ax['sc'])
        plt.pause(0.005)
        
    return range_azimuth, range_elevation, radar_cube, static_clutter

def generate_ra(adc_frame, r_range_id=None, n_range_fft=None, radar_config=None, clutter_removal_enabled=True, method=AoA.mvdr, projection='raw', format='phi', ax=None, is_plot=False):
    if n_range_fft is None:
        n_range_fft = adc_frame.shape[2]

    radar_cube = range_fft(adc_frame, n_range_fft=n_range_fft, window_type_1d=Window.HANNING)    # (n_loop, n_virtual_ant, n_range_bin)
    
    if r_range_id is None:
        r_range_id = {}
        r_range_id['min'] = 0
        r_range_id['max'] = n_range_fft-1
    
    radar_cube = radar_cube[:, :, r_range_id['min']: r_range_id['max']+1]

    angle_estimator = AngleEstimator(radar_config, n_range_fft=n_range_fft, r_range_id=r_range_id)
    # (Optional) Static Clutter Removal
    static_clutter = None
    if clutter_removal_enabled:
        radar_cube, static_clutter = clutter_removal(radar_cube, axis=0)
    # per bin
    radar_cube = np.transpose(radar_cube, (1, 0, 2)) # [n_rx, n_chirp, n_range_fft]
    if method == AoA.fft:
        range_azimuth = angle_estimator.aoa_fft_azimuth(radar_cube, per_range=True)
        if format == 'theta':
            warnings.warn('Warning: Forcing FFT result to be in phi.')
            format = 'phi'
    else:
        range_azimuth = angle_estimator.bf_per_range_azimuth(radar_cube, method=method, format=format)

    if is_plot:
        angle_estimator.plot_range_azimuth_heatmap(range_azimuth, projection=projection, format=format, ax=ax['ra'])
        plot_sc(angle_estimator.dis, static_clutter, ax['sc'])
        plt.pause(0.05)
        
    return range_azimuth, radar_cube, static_clutter

def plot_rd(dis, vel, det_matrix, ax=None):
    if ax is None: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ax.pcolormesh(dis, vel, det_matrix.T)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Doppler velocity (m/s)')

def plot_sc(dis, static_clutter, ax=None):
    static_clutter = 20*np.log10(np.abs(static_clutter))
    if ax is None: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.clear()
    ax.pcolormesh(dis, [i+1 for i in range(static_clutter.shape[0])], static_clutter)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Antenna #')