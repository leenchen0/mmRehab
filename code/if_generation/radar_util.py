import time
import torch
import numpy as np
from enum import Enum

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
TI_RADAR_TYPE   = Enum('TI_RADAR_TYPE', 'iwr1843boost iwr6843aop')
CHIRP_CFG_DICT  = {TI_RADAR_TYPE.iwr1843boost: np.array([0, 2, 1]),
                   TI_RADAR_TYPE.iwr6843aop: np.array([0, 1, 2])}

class Radar:
    """Radar module that defines the location and antenna configuration of the radar (1 transimiter 1 receiver)."""
    cnt = 0
    def __init__(self, Tx_pos=(0, 0, 0), Rx_pos=None, f1=77e9, slope=40e12, ADC_rate=15e6, chirp_time=100e-6, phase_shift=0, noise=None, angles=(0, 0, 0)):
        """
        Parameters:
            Tx_pos: location of the transimitter.
            Rx_pos: location of the receivers. Default to be the same as the transimitter.
            f1: chirp start frequency in Hz.
            slope: chirp slope in Hz/s.
            ADC_rate: ADC sampling rate in Hz.
            chirp_time: the duration of a chirp in seconds.
            phase_shift: apply a phase shift to the antenna, default 0.
            noise: add a Gaussian white noise of power `noise` dB (in relative to the hardware thermal noise) to the simulation. 
            angles: extrinsic rotation angles in degrees in x-y-z. Positive for clockwise rotation. 
        """
        self.f1 = f1
        self.slope = slope
        self.Tx_pos = Tx_pos
        if Rx_pos is not None:
            self.Rx_pos = Rx_pos
        else:
            self.Rx_pos = Tx_pos
        self.Tx_pos = torch.tensor(Tx_pos, dtype=torch.float).to(DEVICE)
        self.Rx_pos = torch.tensor(Rx_pos, dtype=torch.float).to(DEVICE)
        self.rotate(angles)
        self.tx_f = f1
        self.c = 3e8    # speed of light
        self.ADC_rate = ADC_rate
        self.chirp_time = chirp_time
        self.phase_shift = phase_shift
        self.rng = torch.Generator()

        wavelength = self.c / self.f1
        # calculate the hardware thermal noise based on the radar equation
        self.K = 10 ** 2.6 * 1e-3 * wavelength**2 * 1e-2**2 * chirp_time / (4*np.pi)**3 / 4e-21
        self.noise = noise
        if noise is not None:
            # calculate the expected standard deviation of the noise
            self.noise_std = (10**(noise/10)/2)**0.5    # convert dB to linear, calculate std for normal distribution
        # radar equation:
        # snr per chirp = P_tx * G_tx * G_rx * wavelength**2 * radar_cross_section * chirptime / ((4pi)**3 * kT * d**4)
        #               = (12dbm + 7db + 7db) * wavelength**2 * 1e-2**2 * chirptime / ((4pi)**3 * 4e-21 * d**4)
        #               = (10**2.6 * 1e-3) * wavelength**2 * 1e-2**2 * chirptime / ((4pi)**3 * 4e-21 * d**4)
        # http://www.ece.uah.edu/courses/material/EE619-2011/RadarRangeEquation(2)2011.pdf
        # https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/689147/iwr1443-sensing-estimator-snr-calculation
        
        self.name = f'{self.__class__.__name__}_{type(self).cnt}'
        type(self).cnt += 1
        # print(f'[{self.name}] configured to {f1/1e9:.1f} GHz to {(f1+slope*chirp_time)/1e9:.1f} GHz')

    def rotate(self, angles):
        # Convert angles to radians
        angles = torch.tensor(angles, device=DEVICE) * torch.pi / 180

        # Assuming angles are (angle_x, angle_y, angle_z)
        angle_x, angle_y, angle_z = angles

        # Rotation matrices around x, y, z axes
        Rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(angle_x), -torch.sin(angle_x)],
                           [0, torch.sin(angle_x), torch.cos(angle_x)]], device=DEVICE)

        Ry = torch.tensor([[torch.cos(angle_y), 0, torch.sin(angle_y)],
                           [0, 1, 0],
                           [-torch.sin(angle_y), 0, torch.cos(angle_y)]], device=DEVICE)

        Rz = torch.tensor([[torch.cos(angle_z), -torch.sin(angle_z), 0],
                           [torch.sin(angle_z), torch.cos(angle_z), 0],
                           [0, 0, 1]], device=DEVICE)

        # Combined rotation matrix
        RM = torch.matmul(torch.matmul(Rz, Ry), Rx)

        # Check if Rx_pos and Tx_pos are different
        if torch.any(self.Rx_pos != self.Tx_pos):
            v = self.Rx_pos - self.Tx_pos
            v = torch.matmul(RM, v)
            self.Rx_pos = self.Tx_pos + v

    def A(self, d):
        # Assuming self.K is a scalar or a tensor that is on the same device as d
        # Use PyTorch operations instead of NumPy
        return torch.sqrt(self.K / d**4)
    
    def snr_db(self, d):
        """Calculate the snr (dB to thermal noise) of the signal reaching an object at distance `d`"""
        return 10*np.log(self.K/d**4)

    def snr(self, d):
        """Calculate the snr (ratio to thermal noise) of the signal reaching an object at distance `d`"""
        return self.K/d**4

    def distance_between(self, pos1, pos2):
        """Euclidean distance between two points"""
        # Convert pos1 and pos2 to PyTorch tensors if they aren't already
        pos1 = torch.tensor(pos1, device=DEVICE)
        pos2 = torch.tensor(pos2, device=DEVICE)
        # Calculate the Euclidean distance using PyTorch
        dis = torch.norm(pos1 - pos2)
        return dis

    def dis_to(self, pos):
        """The round trip distance between the radar to an object"""
        dis1 = self.calculate_distance(self.Tx_pos, pos)
        dis2 = self.calculate_distance(self.Rx_pos, pos)
        return dis1+dis2
    
    def calculate_distance(self, point, points):
        """Euclidean distance between one point and multiple points"""
        if not isinstance(point, torch.Tensor):
            point = torch.tensor(point, device=DEVICE)
        else:
            point = point.to(DEVICE)

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=DEVICE)
        else:
            points = points.to(DEVICE)
        if points.ndim == 1 and point.ndim == 1:
            return torch.norm(points - point)
        if points.shape[-1] != len(point):
            raise ValueError("The dimensions of the input point and coordinates of other points do not match.")
        distances = torch.norm(points - point, dim=-1)
        return distances

    def IF(self, t, dis):
        """The IF signal resulted from an object"""
        # Perform operations using PyTorch
        tof = dis / self.c
        signal = torch.exp(1j * (2 * np.pi * self.slope * tof * t +
                                 2 * np.pi * self.f1 * tof -
                                 np.pi * self.slope * tof * tof +
                                 self.phase_shift))
        amplitude = self.A(dis / 2)
        return amplitude * signal

    def signal_power(self, X):
        """Get the power of a signal"""
        if not torch.is_tensor(X):
            X = torch.tensor(X, device=DEVICE)
        # Calculate signal power
        return torch.mean(torch.abs(X)**2)
    
    def my_reflect_motion_multi(self, objs: list):
        n_sample = int(np.ceil(self.chirp_time * self.ADC_rate))

        t = torch.arange(0, self.chirp_time, 1 / self.ADC_rate, device=DEVICE)[:n_sample]
        # s = time.perf_counter()
        obj_paths = np.array([obj.get_path() for obj in objs])
        # print(f"[my_reflect_motion_multi]: {time.perf_counter() - s:.2f}s")
        pos = torch.tensor(obj_paths[:, :, 0], device=DEVICE)

        dis = self.dis_to(pos)

        signals = self.IF(t, dis.unsqueeze(2))

        signal = torch.sum(signals, dim=0)


        snr = 10 * torch.log10(torch.stack([self.signal_power(s) for s in torch.unbind(signal, dim=1)]))


        if self.noise is not None:
            # Generate complex noise using PyTorch
            real_noise = torch.normal(0, self.noise_std, size=signal.shape, device=DEVICE)
            imag_noise = torch.normal(0, self.noise_std, size=signal.shape, device=DEVICE)
            noise = real_noise + 1j * imag_noise
            # Update SNR
            snr = snr - 10 * torch.log10(self.signal_power(noise))

            # Add noise to the signal
            signal = signal + noise

        snr = torch.mean(snr)
        info = {'snr': snr.cpu().numpy()}
        return signal.cpu().numpy(), info

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
            radar_config['adc_start_time_sec']  = float(split_words[4]) * 1e-6
            radar_config['ramp_end_time_sec']   = float(split_words[5]) * 1e-6
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
              11 10 09 08
        07 06 05 04 03 02 01 00
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
        radar_config['ant_phase_rot']   = np.ones(12)
    if debug:
        print(f"radar_config: {radar_config}")

    return radar_config

if __name__=="__main__":
    radar_config    = read_config(debug=True)
    num_tx          = radar_config['n_tx']
    slope           = radar_config['freq_slope_hz_sec']
    idle_time       = radar_config['idle_time_sec']
    adc_start_time  = radar_config['adc_start_time_sec']
    ramp_end_time   = radar_config['ramp_end_time_sec']
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