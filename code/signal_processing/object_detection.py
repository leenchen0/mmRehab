import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def threshold(X, th=0.5, db=None):
    """Filter a 2D array with a threshold"""
    if db:
        th = X.max() - db
    else:
        th = X.max() * th
    rows, cols = np.where(X > th)
    res = np.stack((rows, cols, X[rows, cols]), axis=1)
    res = np.asarray(res)
    
    return res

def CFAR_A(data, th=None, win=8, guard=4, debug=False, mode='GO', rank=None, min_db=None, min_rel=None):
    """CFAR Greatest of / Smallest of / Order statistic

    Parameters:
        data: 1d array, should be in dB scale
        th: interpreted as false alarm rate if < 1, a noise threshold in dB otherwise
        win: length of train cells at one side
        guard: length of guard cells at one side
        debug: plot threshold
        mode: GO or SO or OS or CA
        rank: which element to use in OS mode

    Return:
        A list of peak indices
    """
    assert guard < win and "guard cells must be less than train cells"
    data_n = data
    data_n[data_n<0] = 0
    N = win-guard
    res = []
    if mode == 'GO':
        func = lambda x, y, N: np.max((np.sum(x, axis=1), np.sum(y, axis=1)), axis=0)/N
    elif mode == 'SO':
        func = lambda x, y, N: np.min((np.sum(x, axis=1), np.sum(y, axis=1)), axis=0)/N
    elif mode == 'OS':
        if not rank:
            rank = N
        func = lambda x, y, N: np.sort(np.concatenate((x, y), axis=1), axis=1)[:, ::-1][:, N]
    elif mode == 'CA':
        func = lambda x, y, N: (np.sum(x, axis=1) + np.sum(y, axis=1))/N/2
    else:
        raise ValueError(f'Unsupported mode {mode}')

    n = data.shape[0]
    noise = data_n.copy()
    data_n = np.tile(data_n, 7)
    
    noise_buffer    = sliding_window_view(data_n, window_shape=N)
    noise_left      = noise_buffer[n*3-win:n*4-win, :]
    noise_right     = noise_buffer[n*3+guard+1:n*4+guard+1, :]
    try:
        noise = func(noise_left, noise_right, N)
    except Exception as e:
        print(e)
        print(mode, win, guard)
        raise ValueError

    if th is None:
        th = noise
    elif th < 1:        # interpret th as false alarm rate as traditional CFAR
        alpha = N*(th**(-1/N)-1)
        th = noise * alpha
    else:               # interpret th as a noise threshold in dB
        th = noise + th
    
    if min_db is not None:
        maxpower = np.max(data_n)
        minpower = maxpower+min_db
        th[th<minpower] = minpower
    elif min_rel is not None:
        maxpower = np.max(data_n)
        minpower = maxpower * min_rel
        th[th < minpower] = minpower

    res = np.where(data > th)[0]
    if debug:
        plt.plot(data, label='data')
        plt.plot(th, label='th')
        plt.legend()
        plt.show()
    return res

def cfar(X, th=None, win=8, guard=2, mode='OS', min_db=None, min_rel=None, debug=False):
    return CFAR_A(X, th=th, win=win, guard=guard, mode=mode, min_db=min_db, min_rel=min_rel, debug=debug)

def cfar2d(X, th=None, win=8, guard=2, lim=None, debug=False, mode='OS', min_db=None, min_rel=None):
    """2D CFAR detection. 

    Parameters:
        data: a 2D array, e.g. [n_doppler, n_range]
        win: number of neighbour cells to calculate the noise power (each side), including the length of guard cells.
        guard: number of guard cells (each side).
        lim: (optional) if the second dimension is range, then only consider within < lim. 
        debug: draw X and peaks.
        mode: OS (order statistic), CA (cell averaging), GO (greatest of), SO (smallest of).
        min_db: if given and X is in log scale, a peak will only be reported if it is higher than (X.max + min_db).
        min_rel: if given, a peak will only be reported if it is higher than (X.max * min_rel)

    Return:
        [n, 3] array, contains n detected peaks and their indices in the input array, and power.
    """
    if not lim:
        lim = X.shape[1]
    else:
        lim = np.min(X.shape[1], lim)

    bitmask = np.zeros(X.shape[0], dtype=bool)
    res = []
    for k in range(lim):
        cfar1 = cfar(X[:, k], th=th, win=win, guard=guard, mode=mode, min_db=min_db, min_rel=min_rel)   # cfar in doppler domain for each range bin
        bitmask[cfar1] = 1

    detection_list = np.where(bitmask == 1)[0]
    for k in detection_list:
        cfar2 = cfar(X[k], th=th, win=win, guard=guard, mode=mode, min_db=min_db, min_rel=min_rel)  # cfar in range domain for each doppler bin
        for i in cfar2:
            res.append((k, i, X[k, i]))    # save result with (doppler idx, range idx, power)
    res = np.asarray(res)
    if debug:
        Y = np.zeros(X.shape)
        for i, j, _ in res.astype(int):
            Y[i, j] = 1
        _, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].pcolormesh(X)
        axs[1].pcolormesh(Y)
        plt.show()
    return res