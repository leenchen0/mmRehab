import numpy as np
import warnings
import matplotlib.pyplot as plt

from enum import Enum
from scipy import signal
from skimage.feature import peak_local_max

from signal_processing.object_detection import cfar2d
from signal_processing.radar_util import range_freq_to_dis, doppler_freq_to_vel

AoA = Enum('AoA', 'conventional mvdr music fft')    # supported AoA estimation algorithms
Pointcloud_format = Enum('Pointcloud_format', 'spherical cartesian index')

class AngleEstimator():
    def __init__(self, config:dict, n_range_fft=None, n_doppler_fft=None, n_aoa_fft=64, aoa_fft_flag=None, output_size=None, r_range_id=None):
        """
        """
        if n_range_fft is None:
            n_range_fft = config['n_sample']
        if n_doppler_fft is None:
            n_doppler_fft = config['n_loop']
        self.n_range_fft    = n_range_fft
        self.n_doppler_fft  = n_doppler_fft
        self.config         = config.copy()
        self.dis            = range_freq_to_dis(self.config['freq_slope_hz_sec'], self.config['n_sample'], self.config['ADC_rate'], n_range_fft)
        self.vel            = doppler_freq_to_vel(self.config['n_tx'], self.config['n_loop'], self.config['idle_time_sec'],
                                                  self.config['ramp_end_time_sec'], self.config['start_freq_hz'], n_doppler_fft)
        
        if r_range_id is None:
            r_range_id = {}
            r_range_id['min'] = 0
            r_range_id['max'] = n_range_fft-1
        self.dis                = self.dis[r_range_id['min']: r_range_id['max'] + 1]
        self.n_range_fft_cut    = self.dis.shape[0]
        
        self.n_aoa_fft      = n_aoa_fft
        self.aoa_fft_flag   = aoa_fft_flag
        if self.aoa_fft_flag is None:
            if len(self.config['ele_rx']) > 1 and self.config['ele_rx'][1].shape[0] > 2:
                self.aoa_fft_flag = 1       # can use a 2D FFT or two 1D FFTs
            else:
                self.aoa_fft_flag = 0       # use the old style FFT for 1443 type antenna
        self.output_size = output_size

        # prepare FFT parameters
        self.fft_freq_a = np.arange(-1, 1, 2/self.n_aoa_fft)
        self.angle = np.arange(-np.pi/2, np.pi/2, np.pi/self.n_aoa_fft)
        self.angle_phi = np.arcsin(self.fft_freq_a)

        # prepare steering vectors
        n_azimuth = self.config['azi_rx'].shape[0]
        self.azimuth_sv = {
            'phi': self.steering_vector_1d(n_azimuth, self.n_aoa_fft, format='phi'),
            'theta': self.steering_vector_1d(n_azimuth, self.n_aoa_fft, format='theta')
        }
        gs = [8]
        while gs[-1] < n_aoa_fft/4:
            gs.append(gs[-1]*4)
        gs.append(n_aoa_fft)
        # prepare 2d subgrid and fullgrid steering vectors
        self.twod_sv_grid = self.steering_vector_2d_grid(self.config['rx'], gs, format='phi')
        self.twod_sv = self.twod_sv_grid[-1]

    def find_peaks(self, X, th=0.25, n_peaks=1):
        """Find peaks in a signal `X`. 
        
        Parameters:
            X: 1D signal.
            th: threshold in relative to the peak.
            n_peaks: number of peaks to return.
        """
        th = np.max(X)*th
        peaks, vals = signal.find_peaks(X, height=th)
        heights = vals['peak_heights']
        order = np.argsort(heights)[::-1]
        peaks_sorted = peaks[order]   # sort to height descending
        heights_sroted = heights[order]
        if n_peaks is not None:
            peaks_sorted = peaks_sorted[:n_peaks]
            heights_sroted = heights_sroted[:n_peaks]
        
        return peaks_sorted

    def find_peaks_2d(self, X, th=0.25, n_peaks=1):
        """Find peaks in a 2D signal `X`. Applies SRPC.
        
        Parameters:
            X: 2D signal.
            th: threshold in relative to the peak.
            n_peaks: number of peaks to return.
            final: False to disable SRPC or when using in the middle of subgrid search. 
        """
        peaks = peak_local_max(X, threshold_rel=th, num_peaks=n_peaks)
        return peaks
    
    def steering_vector_1d(self, n_rx: int, n_angle: int, theta_range=[-np.pi/2, np.pi/2], format='phi'):
        """Generate a 1D steering vector for a ULA.

        Parameters:
            n_rx: number of rx.
            n_angle: resolution = theta_range/n_angle
            theta_range: default -90 to 90
            format: 'theta' or 'phi'. phi(a) = sin(a)

        Returns:
            [n_rx, n_angle] matrix
        """
        # print(f'Generating {n_angle} steering vectors ({format}).')
        tmin, tmax = theta_range
        if format == 'theta':
            tres = (tmax-tmin)/n_angle
            theta = np.sin(np.arange(tmin, tmax, tres))
        elif format == 'phi':
            tmin, tmax = np.sin(tmin), np.sin(tmax)
            tres = (tmax-tmin)/n_angle
            theta = np.arange(tmin, tmax, tres)
        rx = np.arange(n_rx)
        vec = np.exp(1j*np.outer(rx, np.pi*theta))
        return vec
    
    def steering_vector_2d(self, rx, n_angle: int, theta_range=[-np.pi/2, np.pi/2], format='phi'):
        """Generate a 2D steering vector for a 2D ULA.

        Parameters:
            n_rx: number of rx.
            n_angle: resolution = theta_range/n_angle
            theta_range: default -90 to 90
            format: 'theta' or 'phi'. phi(a) = cos(e)sin(a), phi(e) = sin(e)

        Returns:
            [n_rx, n_elevation, n_azimuth] matrix.
            res[:, i, j] = self.steering_vector_1d_elevation(.., angle[j], .., 'phi')[:, i]  
        """
        tmin, tmax = theta_range
        if format == 'theta':
            tres = (tmax-tmin)/n_angle
            angles = np.arange(tmin, tmax, tres)
            X, Y = np.meshgrid(angles, angles)
            azimuth = np.cos(Y)*np.sin(X)
            elevation = np.sin(Y)                                               # 2 x (n_angle, n_angle)
        elif format == 'phi':
            tmin, tmax = np.sin(tmin), np.sin(tmax)
            tres = (tmax-tmin)/n_angle
            angles = np.arange(tmin, tmax, tres)
            azimuth, elevation = np.meshgrid(angles, angles)                # 2 x (n_angle, n_angle)
        xs = -rx[:, 0]
        zs = -rx[:, 2]

        # azimuth_vec[:, any, i] = self.steering_vector_1d(..)[:, i]
        azimuth_vec = np.exp(1j*np.einsum('i,jk->ijk', xs, np.pi*azimuth))      # (n_rx, n_angle, n_angle)
        elevation_vec = np.exp(1j*np.einsum('i,jk->ijk', zs, np.pi*elevation))  # (n_rx, n_angle, n_angle)

        # res[:, i, j] = self.steering_vector_1d_elevation(.., angle[j])[:, i]
        # assert np.allclose(res[:, ii, jj], self.steering_vector_1d_elevation(rx, angles[jj], n_aoa_fft)[:, ii])
        res = azimuth_vec * elevation_vec
        return res
    
    def steering_vector_2d_grid(self, rx, n_angles, theta_range=[-np.pi/2, np.pi/2], format='phi'):
        """Generate a 2D steering vector grid for a 2D ULA.

        Parameters:
            n_rx: number of rx.
            n_angles: number of angles of each gird, resolution = theta_range/n_angle
            theta_range: default -90 to 90
            format: 'theta' or 'phi'. phi(a) = cos(e)sin(a), phi(e) = sin(e)

        Returns:
            [n_level, n_rx, n_elevation, n_azimuth] matrix.
            res[:, i, j] = self.steering_vector_1d_elevation(.., angle[j], .., 'phi')[:, i]
        """
        res = []
        for n_angle in n_angles:
            sv = self.steering_vector_2d(rx, n_angle, theta_range=theta_range, format=format)
            res.append(sv)
        return res
    
    def xyz_estimate(self, d, wx, wz):
        """Estimate the x-y-z coordinates based on phase and distance

        Parameters:
            d: distance of the point.
            wx: azimuth phase difference between two receivers.
            wz: elevation phase difference between two receivers.
        """
        x = d*wx
        z = d*wz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = np.sqrt(d**2-x**2-z**2)
        return x, y, z
    
    def aoa_fft_1d(self, all_rx, detection):
        """Uses an azimuth FFT first, then an elevation FFT or another azimuth FFT on the second pair of antennas. Suitable for 1843 type antennas."""
        res         = []
        azimuth     = all_rx[self.config['azi_rx']]
        elevation   = all_rx[self.config['ele_rx'][self.aoa_fft_flag]]
        # azimuth fft
        azimuth_fft = np.fft.fft(azimuth, self.n_aoa_fft)
        azimuth_fft = np.fft.fftshift(azimuth_fft)
        azimuth_fft_mag = np.abs(azimuth_fft)
        # find azimuth angles
        p1s = self.find_peaks(azimuth_fft_mag, th=0.5)
        d = self.dis[detection[0]]
        v = self.vel[detection[1]]

        # elevation FFT
        elevation_fft = np.fft.fft(elevation, self.n_aoa_fft)
        elevation_fft = np.fft.fftshift(elevation_fft)
        phase_offset = self.config['phase_offset']
        # find elevation angles for each azimuth angle
        for p1 in p1s:
            wx = self.fft_freq_a[p1]
            if self.aoa_fft_flag == 0:          # use phase for elevation
                wz = np.angle(azimuth_fft[p1]*(elevation_fft[p1].conj())*np.exp(1j*phase_offset*wx*np.pi))/np.pi
                x, y, z = self.xyz_estimate(d, wx, wz)
                res.append((x, y, z, v))
            else:                               # use another fft
                elevation_fft_mag = np.flip(np.abs(elevation_fft))      # flip so that object from top has a positive phase
                p2s = self.find_peaks(elevation_fft_mag, th=0.5)
                for p2 in p2s:
                    wz = self.fft_freq_a[p2]
                    x, y, z = self.xyz_estimate(d, wx, wz)
                    res.append((x, y, z, v))
        return np.array(res).reshape(-1, 4)
    
    def aoa_fft_per_point(self, X, detection_list, return_velocity=True, npass=None):
        """
        self.aoa_fft_2d
        """

        """Performs AoA FFT given FFT data matrix and CFAR detection list, return a point cloud

        Parameters:
            X: [n_virtual_ant, n_range_fft, n_doppler_fft/n_chirps] array.
            detection_list: [n, 3] detection list returned from the 2D CFAR detection.

        Return:
            [n, 4] array that has the object's x-y-z coordinates and velocity.
        """
        n_objs = detection_list.shape[0]
        res = []
        if npass is None:
            npass = 1 if self.aoa_fft_flag == 1 else 2
        if npass == 1 and self.aoa_fft_flag == 0:
            warnings.warn("Warning: calling 2D FFT but receiver row <= 2, forcing to be 1D FFT")
            npass = 2

        if npass == 1:
            func = self.aoa_fft_2d      # 2d FFT
        else:
            func = self.aoa_fft_1d      # old style FFT

        for i in range(n_objs):
            all_rx = X[:, detection_list[i, 0], detection_list[i, 1]]
            res.append(func(all_rx, detection_list[i]))
        res = np.concatenate(res)
        res = res[~np.isnan(res).any(axis=1)]
        if return_velocity:
            return res
        return res[:, :3]
    
    def aoa_fft_azimuth(self, X, per_range=False):
        """Performs azimuth AoA FFT given range FFT data matrix, 
        return an angle power spectrum, summed over all chirps.

        Parameters:
            X: (n_loop, n_virtual_ant, num_range_bins) 3D array.
            per_range: default False, calculate angle power spectrum per range.

        Return:
            [n_angles] 1D array or [n_range, n_angles] if per_range is True.
        """
        azimuth = X[self.config['azi_rx'], :, :]    # [n_rx, n_chirp, n_range_fft]
        azimuth = np.transpose(azimuth, (2, 1, 0))
        azimuth_fft = np.fft.fft(azimuth, self.n_aoa_fft)
        azimuth_fft = np.fft.fftshift(azimuth_fft, axes=2)
        azimuth_fft_mag = np.abs(azimuth_fft)

        if per_range:
            res = np.sum(azimuth_fft_mag, axis=1)
        else:
            res = np.sum(azimuth_fft_mag, axis=(0, 1))
        return res
    
    def diagonal_loading(self, cov):
        """Diagonal loading algorithm on a covariance matrix.
        """
        n = cov.shape[0]
        tr = np.trace(cov)/n
        cov = cov + np.identity(n)*tr*0.05
        return cov
    
    def estimate_n_source(self, eigenvalues, s, debug=False):
        """Minimum description length: estimate the number of data sources given the eigenvalues of the data covariance matrix. 
        
        Parameters:
            eigenvalues: a 1D vector of eigenvalues in ascending order.
            s: number of samples used for calculating the data covariance matrix.

        Return:
            A integer, the estimated number of data sources.
        """
        n = eigenvalues.shape[0]
        s = min(s, 2)
        eigenvalues = eigenvalues/eigenvalues.min()
        aic = np.zeros((n-1))
        mdl = np.zeros((n-1))
        for k in range(n-1):
            t1 = np.product(eigenvalues[k:])**(1/(n-k))
            t2 = np.sum(eigenvalues[k:])/(n-k)
            L = -s*(n-k)*np.log(t1/t2)
            P = k*(2*n-k)
            aic[k] = L + P
            mdl[k] = L + 0.5*P*np.log(s)
        if debug:
            plt.plot(aic, label='aic')
            plt.plot(mdl, label='mdl')
            plt.legend()
            plt.show()
        return np.argmin(mdl)
    
    def nullspace(self, cov, n_sample):
        """Get null space for MUSIC algorithm"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)         # auto sorted in ascending order
        eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
        eigenvectors = np.flip(eigenvectors, axis=1)
        n_obj = self.estimate_n_source(eigenvalues, n_sample)
        if n_obj == 0:
            # warnings.warn('[Warning] zero data source detected')
            return None
        nullvectors = eigenvectors[:, n_obj:]
        nullspace = nullvectors @ np.conj(nullvectors.T)
        return nullspace
    
    def bf(self, cov, steering_vec, method, n_sample=1, cov_inv=None, nullspace=None):
        """Beamforming algorithms for AoA estimation, optimized for speed"""
        res = np.zeros((self.n_aoa_fft))
        if cov_inv is None:
            cov_inv = self.mat_inv(cov)
            if cov_inv is None:
                return res

        if method == AoA.music and nullspace is None:
            nullspace = self.nullspace(cov, n_sample)
            if nullspace is None:
                return res

        # xy = 8x8, a = 512
        if method == AoA.conventional:
            res = np.abs(
                np.einsum('ax,xy,ya->a', np.conj(steering_vec.T), cov, steering_vec))
        elif method == AoA.mvdr:
            res = np.abs(1/np.einsum('ax,xy,ya->a',
                         np.conj(steering_vec.T), cov_inv, steering_vec))
        elif method == AoA.music:
            res = np.abs(1/np.einsum('ax,xy,ya->a',
                         np.conj(steering_vec.T), nullspace, steering_vec))
        return res
    
    def bf_2d(self, cov, steering_vec, method, n_sample=1, cov_inv=None, nullspace=None):
        """Beamforming algorithms for AoA estimation, optimized for speed"""
        if cov_inv is None:
            cov_inv = self.mat_inv(cov)
            if cov_inv is None:
                return

        sv = steering_vec
        svc = np.conj(steering_vec)
        if method == AoA.music:
            if nullspace is None:
                nullspace = self.nullspace(cov, n_sample)
                if nullspace is None:
                    return
        if method == AoA.conventional:
            res = np.abs(np.einsum('xae,xy,yae->ae', svc, cov, sv))
        elif method == AoA.mvdr:
            res = np.abs(1/np.einsum('xae,xy,yae->ae', svc, cov_inv, sv))
        elif method == AoA.music:
            res = np.abs(1/np.einsum('xae,xy,yae->ae', svc, nullspace, sv))
        return res
    
    def bf_2d_grid(self, cov, method, n_sample=1):
        """2D BF using sparser grids then denser grids"""
        sv_grid = self.twod_sv_grid

        eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
        eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
        n_obj = self.estimate_n_source(eigenvalues, n_sample)
        if n_obj == 0:
            return
        cov_inv = self.mat_inv(cov)
        if cov_inv is None:
            return
        nullspace = self.nullspace(cov, n_sample)

        # perform BF at each grid
        peaks = []
        for i, sv in enumerate(sv_grid):
            if i == 0:  # 1st grid
                bf = self.bf_2d(cov, sv, method, 1, cov_inv, nullspace)
                last_gsize = sv.shape[1]
            else:       # define denser grid around peaks
                bf = np.zeros(sv.shape[1:])
                cur_gsize = sv.shape[1]
                ratio = cur_gsize/last_gsize
                rng = int(cur_gsize/last_gsize)+1
                for y, x in peaks:
                    y = int((y+0.5)*ratio)
                    x = int((x+0.5)*ratio)
                    top = max(0, y-rng)
                    bot = min(cur_gsize, y+rng)
                    left = max(0, x-rng)
                    right = min(cur_gsize, x+rng)
                    bf_new = self.bf_2d(cov, sv[:, top:bot, left:right], method, 1, cov_inv, nullspace)
                    bf[top:bot, left:right] = np.maximum(bf[top:bot, left:right], bf_new)
                last_gsize = cur_gsize
            if i == len(sv_grid) - 1:   # final grid
                break
            peaks = self.find_peaks_2d(bf, th=0.90, n_peaks=n_obj)
            if len(peaks) == 0:
                return
        return bf
    
    def mat_inv(self, X):
        """Get inverse matrix"""
        try:
            return np.linalg.inv(X)
        except np.linalg.LinAlgError as err:
            # warnings.warn('[Warning] Singular covariance matrix detected. ')
            return None
        
    def covariance_matrix(self, X, dl=True, fba=True):
        """Calculate only one covariance matrix of a generic input. 

        Parameters:
            data: [n_rx, ...] any input, will be reshaped to [n_rx, -1].
            dl: Perform diagonal loading or not. Required for MVDR.
            fba: Apply Forward-Backward Averaging.

        Returns:
            [n_rx, n_rx] nomalized covariance matrix.
        """
        X = X.reshape((X.shape[0], -1))
        if X.shape[1] > 1:
            X -= X.mean(axis=1)[:, None]
        cov = X @ np.conj(X.T)
        if fba:
            Ex = np.fliplr(np.eye(X.shape[0]))
            cov_bw = Ex @ np.conj(cov) @ Ex
            cov = (cov + cov_bw)/2
        cov /= max(1, X.shape[1]-1)
        if dl:
            cov = self.diagonal_loading(cov)
        return cov
    
    def covariance_matrix_per_range(self, X, dl=True):
        """Calculate the covariance matrix per range bin.

        Parameters:
            X: [n_rx, n_chirp, n_range] 3D array, output from the range-FFT.
            dl: Perform diagonal loading or not. Required for MVDR.

        Returns:
            [n_range, n_rx, n_rx], one covariance matrix for each range bin.
        """
        n_rx, n_chirp, _ = X.shape
        cov = np.zeros((self.n_range_fft_cut, n_rx, n_rx), dtype=complex)
        for d in range(self.n_range_fft_cut):
            cov[d] = self.covariance_matrix(X[:, :, d], dl=dl)
        return cov
    
    def bf_per_range_azimuth(self, X, method=AoA.conventional, format='phi'):
        """Performs azimuth beamforming on each range bin, return a range-Azimuth heatmap.

        Parameters:
            X:[n_rx, n_chirp, n_range_fft] 3D array.
            method: conventional, mvdr, or music.

        Returns:
            [n_range_fft, n_angles] 2D array, the range-Azimuth heatmap.
        """
        azimuth = X[self.config['azi_rx']]
        cov = self.covariance_matrix_per_range(azimuth, dl=True)
        sv = self.azimuth_sv[format]
        res = np.zeros((self.n_range_fft_cut, self.n_aoa_fft))
        for d in range(self.n_range_fft_cut):
            res[d] = self.bf(cov[d], sv, method, azimuth.shape[1])
            # if norm:
            #     res[d] = res[d] * d**4      # to compensate power loss due to further distances
            #     if method == AoA.music:
            #         res[d] = np.log10(res[d]+1)     # music power spectrum has some extremely high values that has to be cut down to log scale
        return res
    
    def bf_per_range_2d(self, X, method=AoA.conventional):
        """Performs 2d beamforming on each range bin, return a point cloud.

        Parameters:
            X:[n_rx, n_chirp, n_range_fft] 3D array.
            method: conventional, mvdr, or music.

        Returns:
            [n, 3] detected point cloud.
        """
        sv = self.twod_sv
        res = np.zeros((self.n_range_fft_cut, self.n_aoa_fft, self.n_aoa_fft))
        for i in range(X.shape[2]):
            all_rx = X[:, :, i]           # (12, 50)
            cov = self.covariance_matrix(all_rx)
            res[i] = self.bf_2d(cov, sv, method)
            # res[i] = self.bf_2d_grid(cov, method, n_sample=all_rx.shape[1])
        return res
    
    def bf_per_range_static(self, X, method=AoA.conventional, format=Pointcloud_format.cartesian):
        """Performs 2d beamforming on each range bin, return a point cloud.

        Parameters:
            X:[n_rx, n_range_fft] 3D array.
            method: conventional, mvdr, or music.

        Returns:
            [n, 3] detected point cloud.
        """
        X   = X[:, np.newaxis, :]
        res = np.zeros((self.n_range_fft_cut, self.n_aoa_fft, self.n_aoa_fft))
        range_angles = np.zeros((self.n_aoa_fft*self.n_aoa_fft, self.n_range_fft_cut))
        for i in range(self.n_range_fft_cut):
            all_rx = X[:, :, i]           # (12, 50)
            cov = self.covariance_matrix(all_rx)
            eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
            eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
            n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
            if n_obj == 0:
                continue

            # bf = self.bf_2d(cov, sv, method)
            bf = self.bf_2d_grid(cov, method, n_sample=all_rx.shape[1]) # (azi, ele)
            if bf is None:
                continue
            range_angle = bf.T.reshape(-1)

            res[i] = bf
            range_angles[:, i] = range_angle

        range_angles[range_angles<np.max(range_angles)*0.2] = 0
        det_res = cfar2d(range_angles, guard=5, win=10, min_rel=0.9, debug=False)
        
        if det_res.shape[0] == 0:
            return res, np.zeros((0, 4))
        wx  = (det_res[:, 0] % self.n_aoa_fft).astype(int)
        wz  = (np.floor(det_res[:, 0] / self.n_aoa_fft)).astype(int)
        d_i = (det_res[:, 1]).astype(int)
        pc = np.zeros((d_i.shape[0], 4))
        if format == Pointcloud_format.cartesian:
            d  = self.dis[d_i]
            pc[:, 0] = d*self.fft_freq_a[wx]
            pc[:, 2] = d*self.fft_freq_a[wz]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pc[:, 1] = np.sqrt(d**2-pc[:, 0]**2-pc[:, 2]**2)
        elif format == Pointcloud_format.spherical:
            pc[:, 0] = self.dis(d_i)
            pc[:, 1] = self.angle_phi(wx)
            pc[:, 2] = self.angle_phi(wz)
            # pc = np.concatenate((self.dis(d_i)[:, np.newaxis], self.angle_phi(wx)[:, np.newaxis], self.angle_phi(wz)[:, np.newaxis], np.zeros((x.shape[0], 1))), axis=1)
        elif format == Pointcloud_format.index:
            pc[:, 0] = d_i
            pc[:, 1] = wx
            pc[:, 2] = wz
            # pc = np.concatenate((d_i[:, np.newaxis], wx[:, np.newaxis], wz[:, np.newaxis], np.zeros((x.shape[0], 1))), axis=1)
        # pc = []
        # for i in range(det_res.shape[0]):
        #     wx = int(det_res[i, 0] % self.n_aoa_fft)
        #     wz = int(np.floor(det_res[i, 0] / self.n_aoa_fft))
        #     d  = int(det_res[i, 1])
        #     if format == Pointcloud_format.cartesian:
        #         x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
        #         pc.append((x, y, z, 0))
        #     elif format == Pointcloud_format.spherical:
        #         pc.append((d, self.angle_phi(wx), self.angle_phi(wz), 0))
        #     elif format == Pointcloud_format.index:
        #         pc.append((i, wx, wz, 0))
        pc = np.asarray(pc).reshape((-1, 4))
        pc = pc[~np.isnan(pc).any(axis=1)]       # remove points with nan coordinates
        # print(f'Beamforming finished with {res.shape[0]} points.')
        res[np.isnan(res)] = 0
        return res, pc
    
    def bf_per_point_elevation(self, X, detection_list, method=AoA.conventional, format=Pointcloud_format.index):
        """Performs elevation beamforming given a dection list in range-azimuth domain, 
        return their x-y-z coordinates. Used for non-Doppler DPC.
        Format has to be phi.

        Parameters:
            X: [n_rx, n_chirp, n_range_fft] 3D array.
            detection_list: [n, 2] detection list, (range, wx), returned from the 2D CFAR detection.
            method: conventional, mvdr, or music.

        Returns:
            [n, 3] array that has the object's x-y-z coordinates.
        """
        n_objs = detection_list.shape[0]
        sv = self.twod_sv
        res = []
        # tmp = []
        for i in range(n_objs):
            all_rx  = X[:, :, int(detection_list[i, 0])]           # (12, 50)
            wx      = int(detection_list[i, 1])
            cov     = self.covariance_matrix(all_rx)

            sv = self.twod_sv[:, :, wx]
            bf = self.bf(cov, sv, method, n_sample=X.shape[1])
            if bf is None:
                continue

            eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
            eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
            n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
            # tmp.append(n_obj)
            if n_obj == 0:
                continue

            peaks = self.find_peaks(bf, n_peaks=n_obj)
            d = self.dis[int(detection_list[i, 0])]
            for wz in peaks:
                if format == Pointcloud_format.cartesian:
                    x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                    res.append((x, y, z))
                elif format == Pointcloud_format.spherical:
                    res.append((d, self.angle_phi(wx), self.angle_phi(wz)))
                elif format == Pointcloud_format.index:
                    res.append((detection_list[i, 0], wx, wz))
        res = np.asarray(res).reshape((-1, 3))
        res = res[~np.isnan(res).any(axis=1)]       # remove points with nan coordinates
        # print(f'Beamforming finished with {res.shape[0]} points.')
        # print(tmp)
        return res
    
    def bf_per_point_doppler(self, X, detection_list, format=Pointcloud_format.index):
        """Performs doppler fft given a dection list in range-azimuth-elevation domain, 
        return their x-y-z coordinates.
        Format has to be phi.

        Parameters:
            X: [n_rx, n_chirp, n_range_fft] 3D array.
            detection_list: [n, 3] detection list, (range, wx, wz), returned from the 2D CFAR detection.
            method: conventional, mvdr, or music.

        Returns:
            [n, 3] array that has the object's x-y-z coordinates.
        """
        n_objs  = detection_list.shape[0]
        sv      = self.twod_sv
        res     = []
        for i in range(n_objs):
            all_rx  = X[:, :, int(detection_list[i, 0])]           # (12, 50)
            d       = self.dis[int(detection_list[i, 0])]
            wx      = int(detection_list[i, 1])
            wz      = int(detection_list[i, 2])
            cov     = self.covariance_matrix(all_rx)

            # bf          = self.bf_2d(cov, sv, method)
            weight      = np.dot(self.mat_inv(cov), sv[:, wz, wx])
            fft_input   = np.dot(np.conj(weight), all_rx)
            doppler_fft = np.fft.fftshift(np.fft.fft(fft_input))

            v_i = np.argmax(np.abs(doppler_fft))
            v   = self.vel[v_i]

            if format == Pointcloud_format.cartesian:
                x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                res.append((x, y, z, v))
            elif format == Pointcloud_format.spherical:
                res.append((d, self.angle_phi(wx), self.angle_phi(wz), v))
            elif format == Pointcloud_format.index:
                res.append((detection_list[i, 0], wx, wz, v_i))
        
        res = np.asarray(res).reshape((-1, 4))
        res = res[~np.isnan(res).any(axis=1)]       # remove points with nan coordinates
        return res
    
    def generate_point_cloud_with_doppler(self, rd_profile, peaks, method=AoA.fft, npass=2):
        """
        self.bf_per_point_2d
        self.bf_per_point
        """

        """Generate a point cloud using the TI OOB demo algorithm

        Parameters:
            rd_profile: Range-Doppler profile.(n_virtual_ant, n_range_bin, n_doppler_bin) 
                        ADC data reorganized by vrx instead of physical rx.
            peaks: [n, 3] detection list returned from algorithm of object detection.
            method: AoA estimation algorithm to use.
            npass: one pass algorithm (azimuth and elevation together) or two pass algorithm (azimuth then elevation).

        Return:
            (n, 3) point cloud.
        """
        # AoA estimation for each range-Doppler peak
        if method == AoA.fft:
            point_cloud = self.aoa_fft_per_point(rd_profile, peaks, npass=npass)
        else:
            if npass == 1:
                func = self.bf_per_point_2d
            elif npass == 2:
                func = self.bf_per_point
            else:
                raise ValueError(f"npass value {npass} incorrect")
            point_cloud = func(rd_profile, peaks, method=method)
        if self.output_size is not None:
            point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], self.output_size, replace=False)]
        return point_cloud
    
    def plot_range_azimuth_heatmap(self, range_azimuth, projection='raw', format='phi', ax=None):
        """Plot a 2D range-azimuth heatmap
        
        Parameters:
            range_azimuth: range-azimuth heatmap.
            projection: `raw`, `polar` or `cartesian`
            format: `phi` or `theta`
        """
        # range_azimuth = np.log2(range_azimuth)
        range_azimuth = range_azimuth/range_azimuth.max()
        print(f'Plotting range-azimuth heatmap in {format} domain.', end='\r')
        if format == 'phi':
            azimuth = self.angle_phi
        else:
            azimuth = self.angle

        bg = plt.cm.get_cmap(plt.rcParams["image.cmap"])(0)

        if projection == 'raw':
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            ax.clear()
            ax.pcolormesh(azimuth/np.pi*180, self.dis, range_azimuth)
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Range (m)')
        elif projection == 'polar':
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='polar')
            ax.clear()
            ax.pcolormesh(azimuth, self.dis, range_azimuth)
            ax.set_thetamin(-75)
            ax.set_thetamax(75)
            ax.set_theta_direction(-1)
            ax.grid(True, alpha=0.2)
            ax.set_theta_zero_location('N')
            ax.set_facecolor(bg)
            # cb = fig.colorbar(pc)
            # cb.ax.tick_params(labelsize=16)
        elif projection == 'cartesian':
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            ax.clear()
            dis, angle = np.meshgrid(self.dis, azimuth)
            xs = dis*np.cos(angle)
            ys = dis*np.sin(angle)
            ax.pcolormesh(ys, xs, range_azimuth.T)
            ax.set_facecolor(bg)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')