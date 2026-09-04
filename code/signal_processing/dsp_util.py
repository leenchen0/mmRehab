import numpy as np

try:
    from enum import Enum
except ImportError:
    print("enum only exists in Python 3.4 or newer")

try:
    class Window(Enum):
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4
except NameError:
    class Window:
        BARTLETT = 1
        BLACKMAN = 2
        HAMMING  = 3
        HANNING  = 4

def windowing(input, window_type, axis=0):
    """Window the input based on given window type.

    Args:
        input: input numpy array to be windowed.

        window_type: enum chosen between Bartlett, Blackman, Hamming, Hanning and Kaiser.

        axis: the axis along which the windowing will be applied.
    
    Returns:

    """
    window_length = input.shape[axis]
    if window_type == Window.BARTLETT:
        window = np.bartlett(window_length)
    elif window_type == Window.BLACKMAN:
        window = np.blackman(window_length)
    elif window_type == Window.HAMMING:
        window = np.hamming(window_length)
    elif window_type == Window.HANNING:
        window = np.hanning(window_length)
    else:
        raise ValueError("The specified window is not supported!!!")

    output = input * window

    return output