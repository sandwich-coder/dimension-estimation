from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np
from scipy import linalg as la
from scipy.fft import fftn as ft, ifftn as ift, fftfreq, fftshift
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad
import matplotlib as mpl
from matplotlib import pyplot as pp

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# Spatial median is set as the point of rotation.
# There is a well-known algorithm dedicated for computing spatial median, but the 'minimize' is used for quick experimentation.
def median(X):
    if type(X) != np.ndarray:
        raise TypeError('Input must be a \'numpy.ndarray\'.')
    if X.dtype != np.float64:
        X = X.astype('float64')
    if X.ndim != 2:
        raise ValueError('The shape is not the dataset standard.')

    def _distance_sum(y):
        y = y.reshape([1, y.shape[0]])
        distances = cdist(y, X)
        return distances.sum()

    initial = X.mean(axis = 0)
    result = minimize(_distance_sum, initial, method = 'Nelder-Mead')

    return result.x


def dispersion(X):
    if type(X) != np.ndarray:
        raise TypeError('Input must be a \'numpy.ndarray\'.')
    if X.dtype != np.float64:
        X = X.astype('float64')
    if X.ndim != 2:
        raise ValueError('The shape is not the dataset standard.')

    #centered
    X = X - median(X)

    def _func(row1, row2):
        return np.median(row1 * row2, axis = 0)

    spread = cdist(X.transpose(), X.transpose(), metric = _func)
    return spread



# - test -

#points
data = np.stack([
    np.arange(10, dtype = 'float64'), np.arange(10, dtype = 'float64')
    ], axis = 1)
outlier = np.array([
    [10, -100],
    [6, 100]
    ], dtype = 'float64')
data = np.concatenate([data, outlier], axis = 0)

#median
med = median(data)

spread = dispersion(data)
eig, eigmat = la.eigh(spread)
eig, eigmat = np.flip(eig), np.flip(eigmat)
rotated = data @ eigmat


# - plot -

fig = pp.figure(layout = 'constrained', figsize = (11, 5))
fig.suptitle('comparison')
gs = fig.add_gridspec(nrows = 1, ncols = 2, hspace = 0.2)

ax_1 = fig.add_subplot(gs[0])
ax_1.set_title('original')
ax_1.set_xlabel('x')
ax_1.set_ylabel('y')
ax_1.set_box_aspect(1)
ax_1.set_aspect('equal')
pp.setp(ax_1.get_xticklabels(), rotation = 30, ha = 'right', rotation_mode = 'anchor')
pp.setp(ax_1.get_yticklabels(), rotation = 60, ha = 'right', rotation_mode = 'anchor')
plot_1_1 = ax_1.plot(
        data[:, 0], data[:, 1],
        marker = '.', markersize = 1,
        linestyle = '',
        color = 'blue',
        label = 'data'
        )
plot_1_2 = ax_1.plot(
        med[0], med[1],
        marker = '+', markersize = 5,
        linestyle = '',
        color = 'red',
        label = 'median \n(rotation point)'
        )

ax_2 = fig.add_subplot(gs[1])
ax_2.set_title('aligned')
ax_2.set_xlabel('x\'')
ax_2.set_ylabel('y\'')
ax_2.set_box_aspect(1)
ax_2.set_aspect('equal')
pp.setp(ax_2.get_xticklabels(), rotation = 30, ha = 'right', rotation_mode = 'anchor')
pp.setp(ax_2.get_yticklabels(), rotation = 60, ha = 'right', rotation_mode = 'anchor')
plot_2 = ax_2.plot(
        rotated[:, 0], rotated[:, 1],
        marker = '.', markersize = 1,
        linestyle = '',
        color = 'green'
        )


ax_1.legend()
pp.show()
