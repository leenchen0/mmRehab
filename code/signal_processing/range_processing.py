import numpy as np
from signal_processing.dsp_util import windowing

def range_fft(adc_data, n_range_fft=None, window_type_1d=None, axis=-1):
    """Perform 1D FFT on complex-format ADC data.

    Perform optional windowing and 1D FFT on the ADC data.

    Args:
        adc_data (ndarray): (n_loop, n_virtual_ant, n_sample). Performed on each frame. adc_data
                            is in complex by default. Complex is float32/float32 by default.
        window_type_1d (dsp_utils.Window): Optional window type on 1D FFT input. Default is None. Can be selected
                                                from Bartlett, Blackman, Hanning and Hamming.
    
    Returns:
        radar_cube (ndarray): (n_loop, n_virtual_ant, num_range_bins). Also called fft_1d_out
    """
    if n_range_fft is None:
        n_range_fft = adc_data.shape[axis]

    # Note: np.fft.fft is a 1D operation, using higher dimension input defaults to slicing last axis for transformation
    # windowing numA x numB suggests the coefficients is numA-bits while the 
    # input and output are numB-bits. Same rule applies to the FFT.
    fft1d_window_type = window_type_1d
    if fft1d_window_type:
        fft1d_in = windowing(adc_data, fft1d_window_type, axis=axis)
    else:
        fft1d_in = adc_data

    # Note: np.fft.fft is a 1D operation, using higher dimension input defaults to slicing last axis for transformation
    radar_cube = np.fft.fftshift(np.fft.fft(fft1d_in, n=n_range_fft, axis=axis), axes=axis)

    return radar_cube