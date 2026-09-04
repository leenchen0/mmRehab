import numpy as np
from signal_processing.dsp_util import windowing

def doppler_fft(radar_cube, n_doppler_fft=None, window_type_2d=None):
    """Perform 2D FFT on the radar_cube.

    Args:
        radar_cube (ndarray): Output of the 1D FFT. It has the shape of (n_loop, n_virtual_ant, n_range_bin). 
        window_type_2d (dsp_utils.Window): Optional windowing type before doppler FFT.
    
    Returns:
        detMatrix (ndarray): (n_range_bin, n_doppler_bin) complete range-dopper information. Original datatype is
                             uint16_t. Note that azimuthStaticHeatMap can be extracted from zero-doppler index for
                             visualization.
        aoa_input (ndarray): (n_range_bin, n_virtual_ant, n_doppler_bin) ADC data reorganized by vrx instead of
                             physical rx.
    """
    if n_doppler_fft is None:
        n_doppler_fft = radar_cube.shape[0]

    fft2d_in = radar_cube

    # transpose to (n_range_bin, n_virtual_ant, n_loop)
    fft2d_in = np.transpose(fft2d_in, axes=(2, 1, 0))

    # Windowing 16x32
    if window_type_2d:
        fft2d_in = windowing(fft2d_in, window_type_2d, axis=2)

    # It is assumed that doppler is at the last axis.
    # FFT 32x32
    fft2d_out   = np.fft.fftshift(np.fft.fft(fft2d_in, n=n_doppler_fft), axes=2)
    rd_profile  = fft2d_out

    # Save zero-Doppler as azimuthStaticHeatMap, watch out for the bit shift in
    # original code.

    # Log_2 Absolute Value
    fft2d_log_abs_sum = np.sum(np.abs(fft2d_out), axis=1)
    # fft2d_log_abs_sum = 20*np.log10(np.sum(np.abs(fft2d_out), axis=1))
    # fft2d_log_abs_sum = np.sum(20*np.log10(np.abs(fft2d_out)), axis=1)
    # fft2d_log_abs_sum = fft2d_log_abs_sum - np.mean(fft2d_log_abs_sum, axis=0)

    return fft2d_log_abs_sum, rd_profile
    
def fine_motion(static_clutters):
    tmp = np.sum(static_clutters, axis=1)
    tmp = tmp - np.mean(tmp, axis=0)
    fine_doppler = np.fft.fftshift(np.fft.fft(tmp, axis=0), axes=0)

    fine_doppler = np.abs(fine_doppler)
    return fine_doppler

def fine_motion_1(static_clutters):
    static_clutters = static_clutters - np.mean(static_clutters, axis=0)[np.newaxis, :, :]
    fine_doppler = np.fft.fftshift(np.fft.fft(static_clutters, axis=0), axes=0)

    # fine_doppler = np.abs(np.sum(fine_doppler, axis=1))
    # fine_doppler = np.sum(np.abs(fine_doppler, axis=1))

    return np.abs(np.sum(fine_doppler, axis=1)), fine_doppler

def clutter_removal(input_val, axis=0):
    """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.

    Args:
        input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
            e.g. [n_loop, n_virtual_ant, n_range_bin], it is applied along the first axis.
        axis (int): Axis to calculate mean of pre-doppler.

    Returns:
        ndarray: Array with static clutter removed.
        mean: Array with static clutter
    """
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean

    return output_val.transpose(reordering), mean