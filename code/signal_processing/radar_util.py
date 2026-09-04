import numpy as np
from enum import Enum

TI_RADAR_TYPE       = Enum('TI_RADAR_TYPE', 'iwr1843boost iwr6843aop arbe vayyar')
CHIRP_CFG_DICT      = {TI_RADAR_TYPE.iwr1843boost: np.array([0, 1, 2]),
                       TI_RADAR_TYPE.iwr6843aop: np.array([0, 1, 2])}
REORDEING           = {TI_RADAR_TYPE.iwr1843boost: np.array([ 0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7]),
                       TI_RADAR_TYPE.iwr6843aop: np.array([3, 1, 2, 0, 11, 9, 7, 5, 10, 8, 6, 4])}
ANT_PHASE_ROTATE    = {TI_RADAR_TYPE.iwr1843boost: np.ones(12),
                       TI_RADAR_TYPE.iwr6843aop: np.array([-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])}

def cal_center_freq(start_freq, adc_start_time, slope, num_sample, ADC_rate):
    return start_freq + adc_start_time * slope + cal_bandwidth(slope, num_sample, ADC_rate)/2

def cal_chirp_period(num_tx, idle_time, ramp_end_time):
    return num_tx * (idle_time + ramp_end_time)

def cal_chirp_time(num_sample, ADC_rate):
    return num_sample / ADC_rate

def cal_bandwidth(slope, num_sample, ADC_rate):
    chirp_time = cal_chirp_time(num_sample, ADC_rate)
    return slope * chirp_time

def cal_max_vel(num_tx, idle_time, ramp_end_time, center_freq):
    chirp_period = cal_chirp_period(num_tx, idle_time, ramp_end_time)
    return 3e8/(4*chirp_period*center_freq)

def cal_vel_resolution(num_tx, num_loop, idle_time, ramp_end_time, center_freq):
    max_vel = cal_max_vel(num_tx, idle_time, ramp_end_time, center_freq)
    return (2*max_vel)/num_loop

def cal_ran_resolution(slope, num_sample, ADC_rate):
    bandwidth = cal_bandwidth(slope, num_sample, ADC_rate)
    return 3e8/(2*bandwidth)

def range_freq_to_dis(slope, num_sample, ADC_rate, num_range_fft):
    ran_resolution = cal_ran_resolution(slope, num_sample, ADC_rate)
    return np.arange(-num_range_fft/2, num_range_fft/2) * (num_sample / num_range_fft) * ran_resolution

def doppler_freq_to_vel(num_tx, num_loop, idle_time, ramp_end_time, center_freq, num_doppler_fft):
    vel_resolution = cal_vel_resolution(num_tx, num_loop, idle_time, ramp_end_time, center_freq)

    return np.arange(-num_doppler_fft/2, num_doppler_fft/2) * (num_loop / num_doppler_fft) * vel_resolution

def read_config(config_file='./config_files/iwr1843boost.cfg', layout='iwr1843boost', debug=False):
    radar_config = {}
    config = [line.rstrip('\r\n') for line in open(config_file)]

    for i in config:
        # Split the line
        split_words = i.split(" ")

        # Get the information about the profile configuration
        if "profileCfg" in split_words[0]:
            radar_config['start_freq_hz']       = float(split_words[2]) * 1e9
            radar_config['idle_time_sec']       = float(split_words[3]) * 1e-6
            radar_config['ramp_end_time_sec']   = float(split_words[5]) * 1e-6
            radar_config['adc_start_time_sec']  = float(split_words[4]) * 1e-6
            radar_config['freq_slope_hz_sec']   = float(split_words[8]) * 1e12
            radar_config['ADC_rate']            = float(split_words[11]) * 1e3
            radar_config['n_sample']            = int(split_words[10])

        elif 'channelCfg' in split_words[0]:
            radar_config['n_rx']        = bin(int(split_words[1])).count('1')
            radar_config['n_tx']        = bin(int(split_words[2])).count('1')
            radar_config['chirp_cfg']   = np.empty(radar_config['n_tx'])

        elif 'chirpCfg' in split_words[0]:
            radar_config['chirp_cfg'][int(split_words[1])] = np.log2(int(split_words[8]))
            
        elif "frameCfg" in split_words[0]:
            radar_config['n_loop']  = int(split_words[3])
            radar_config['fps_s']   = int(split_words[5]) * 1e-3

    if layout == TI_RADAR_TYPE.iwr1843boost:
        """
              08 09 10 11
        00 01 02 03 04 05 06 07
        """
        xs1 = np.arange(0, -8, -1)
        xs2 = np.arange(-2, -6, -1)
        ys1 = np.zeros(8)
        ys2 = np.zeros(4)
        zs1 = np.zeros(8) - 1
        zs2 = np.zeros(4)

        azimuth                         = np.array((xs1, ys1, zs1))
        elevation                       = np.array((xs2, ys2, zs2))
        radar_config['rx']              = np.concatenate((azimuth, elevation), axis=-1).T
        radar_config['azi_rx']          = np.arange(0, 8, dtype=int)        # azimuth rx is the first row
        radar_config['ele_rx']          = [np.arange(8, 12, dtype=int)]     # elevation rx is the second row
        radar_config['phase_offset']    = 2
        # radar_config['ant_phase_rot']   = np.ones(12)
    elif layout == TI_RADAR_TYPE.iwr6843aop:
        """
        00 01
        02 03
        04 05 06 07
        08 09 10 11
        """
        ant_geometry0                   = np.array([[0, -1,  0, -1,  0, -1, -2, -3,  0, -1, -2, -3]])
        ant_geometry1                   = np.array([[0,  0, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3]])
        radar_config['rx']              = np.concatenate((ant_geometry0, np.zeros((1, 12)), ant_geometry1), axis=0).T
        radar_config['azi_rx']          = np.arange(8, 12, dtype=int)
        radar_config['ele_rx']          = [np.array([4, 5, 6, 7]), np.array([8, 4, 2, 0])]
        radar_config['phase_offset']    = 0
        # radar_config['ant_phase_rot']   = np.array([-1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1])
    elif layout == TI_RADAR_TYPE.arbe:
        """
        00     01 ...   47
        48     49 ...   95
        ...
        2256 2257 ... 2303
        """
        radar_config['n_rx']            = 48
        radar_config['n_tx']            = 48
        ant_geometry0                   = np.tile(np.arange(0, -48, -1), 48).reshape(1, 48*48)
        ant_geometry1                   = np.tile(np.arange(0, -48, -1), (48, 1)).T.reshape(1, 48*48)
        radar_config['rx']              = np.concatenate((ant_geometry0, np.zeros((1, 48*48)), ant_geometry1), axis=0).T
        radar_config['azi_rx']          = np.arange(0, 48, dtype=int)
        radar_config['ele_rx']          = [np.arange(48*47, 48*48, dtype=int), 48*np.arange(0, 48,dtype=int)]
        radar_config['phase_offset']    = 0
    elif layout == TI_RADAR_TYPE.vayyar:
        """
        00   01 ...  23
        24   49 ...  95
        ...
        504 505 ... 527
        """
        radar_config['n_rx']            = 22
        radar_config['n_tx']            = 24
        ant_geometry0                   = np.tile(np.arange(0, -24, -1), 22).reshape(1, 24*22)
        ant_geometry1                   = np.tile(np.arange(0, -24, -1), (22, 1)).T.reshape(1, 24*22)
        radar_config['rx']              = np.concatenate((ant_geometry0, np.zeros((1, 24*22)), ant_geometry1), axis=0).T
        radar_config['azi_rx']          = np.arange(0, 24, dtype=int)
        radar_config['ele_rx']          = [np.arange(24*21, 24*22, dtype=int), 24*np.arange(0, 22, dtype=int)]
        radar_config['phase_offset']    = 0
    if debug:
        print(f"radar_config: {radar_config}")

    return radar_config

if __name__=="__main__":
    radar_config = read_config(debug=True)
    num_tx          = radar_config['n_tx']
    slope           = radar_config['freq_slope_hz_sec']
    idle_time       = radar_config['idle_time_sec']
    ramp_end_time   = radar_config['ramp_end_time_sec']
    adc_start_time  = radar_config['adc_start_time_sec']
    center_freq     = radar_config['start_freq_hz']
    num_chirp       = radar_config['n_loop']
    num_sample      = radar_config['n_sample']
    ADC_rate        = radar_config['ADC_rate']
    start_freq      = radar_config['start_freq_hz']

    print(f"Chirp Repetition Period(us): {cal_chirp_period(num_tx, idle_time, ramp_end_time) * 1e6}")
    print(f"Maximum Velocity (m/s): {cal_max_vel(num_tx, idle_time, ramp_end_time, center_freq)}")
    print(f"Velocity Resolution (m/s): {cal_vel_resolution(num_tx, num_chirp, idle_time, ramp_end_time, center_freq)}")
    print(f"Chirp Time (us): {cal_chirp_time(num_sample, ADC_rate)*1e6}")
    print(f"Bandwidth (MHz): {cal_bandwidth(slope, num_sample, ADC_rate)*1e-6}")
    print(f"Range Resolution (m): {cal_ran_resolution(slope, num_sample, ADC_rate)}")
    print(f"Center Frequency (GHz): {cal_center_freq(start_freq, adc_start_time, slope, num_sample, ADC_rate)*1e-9}")