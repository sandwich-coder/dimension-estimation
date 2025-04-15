from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np
from scipy import linalg as la
from scipy.fft import fftn, ifftn, fftfreq, fftshift
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad
import matplotlib as mpl
from matplotlib import pyplot as pp

import pandas as pd
from sklearn.decomposition import PCA
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


# This function is continuous but not positive-semidefinite.
# According to Mercer's theorem, only matrix constructed by continuous PSD function retains the information under diagonalization.
# The PCA is one notable example of such functions.
# Yet, diagonalization of this matrix still seems to "align".
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


class BareEstimator:
    def __repr__(self):
        return 'bare estimator'

    def __call__(
            self,
            X,
            width,
            batch_count = 1000,
            exact = False
            ):
        if type(X) != np.ndarray:
            raise TypeError('Input must be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The shape must be the dataset standard.')
        if not isinstance(width, (int, float)):
            raise TypeError('Tile width should be an integer or float.')
        if width <= 0:
            raise ValueError('Tile width must be positive.')
        if not isinstance(batch_count, int):
            raise TypeError('\'batch_count\' should be an integer.')
        if batch_count < 1:
            raise ValueError('\'batch_count\' must be positive.')
        if not isinstance(exact, bool):
            raise TypeError('\'exact\' should be boolean.')


        #quantized
        tile = X / width - np.float64(0.5)
        tile = tile.round().astype('int64')
        tile = np.unique(tile, axis = 0)


        # - counted -

        batch = copy(np.array_split(tile, batch_count, axis = 0))
        if batch_count > tile.shape[0]:
            batch = batch[:tile.shape[0]]

        adjacency = []
        for lll in tqdm(batch, colour = 'magenta'):

            _ = lll.reshape([lll.shape[0], 1, lll.shape[1]])
            compared = _.repeat(tile.shape[0], axis = 1)

            _ = tile.reshape([tile.shape[0], 1, tile.shape[1]])
            _ = _.swapaxes(0, 1)
            all_ = _.repeat(compared.shape[0], axis = 0)

            distance = np.max(
                    np.absolute(compared - all_),
                    axis = 2
                    )
            is_adjacent = distance == 1

            adjacency_batch = is_adjacent.astype('int64')
            adjacency_batch = adjacency_batch.sum(axis = 1, dtype = 'int64')
            adjacency.append(adjacency_batch)

        adjacency = np.concatenate(adjacency, axis = 0)


        dimension = np.log(np.median(adjacency, axis = 0) + 1, dtype = 'float64') / np.log(3, dtype = 'float64')
        if exact:
            dimension = dimension.tolist()
        else:
            dimension = dimension.round().astype('int64')
            dimension = dimension.tolist()
        return dimension





' ========== test ========== ' """


#sample points
data = np.stack([
    np.arange(10, dtype = 'float64'), np.arange(10, dtype = 'float64')
    ], axis = 1)
outlier = np.array([
    [10, -100]
    ], dtype = 'float64')
data = np.concatenate([data, outlier], axis = 0)

#centers
mean = data.mean(axis = 0, dtype = 'float64')
med = median(data)

#std-rotation
pca = PCA(svd_solver = 'full')
pca.fit(data)
std_rotated = pca.transform(data)

#MAD-rotation
spread = dispersion(data)
eig, eigmat = la.eigh(spread)
eig, eigmat = np.flip(eig), np.flip(eigmat)
mad_rotated = data @ eigmat


# - plot -

fig = pp.figure(layout = 'constrained', figsize = (17, 5))
fig.suptitle('comparison')
gs = fig.add_gridspec(nrows = 1, ncols = 3, hspace = 0.2)

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
        mean[0], mean[1],
        marker = 'x', markersize = 6,
        linestyle = '',
        color = 'grey',
        label = 'mean'
        )
plot_1_3 = ax_1.plot(
        med[0], med[1],
        marker = '+', markersize = 6,
        linestyle = '',
        color = 'red',
        label = 'median'
        )
ax_1.legend()

ax_2 = fig.add_subplot(gs[1])
ax_2.set_title('std-aligned')
ax_2.set_xlabel('x\'')
ax_2.set_ylabel('y\'')
ax_2.set_box_aspect(1)
ax_2.set_aspect('equal')
pp.setp(ax_2.get_xticklabels(), rotation = 30, ha = 'right', rotation_mode = 'anchor')
pp.setp(ax_2.get_yticklabels(), rotation = 60, ha = 'right', rotation_mode = 'anchor')
plot_2 = ax_2.plot(
        std_rotated[:, 0], std_rotated[:, 1],
        marker = '.', markersize = 1,
        linestyle = '',
        color = 'grey'
        )

ax_3 = fig.add_subplot(gs[2])
ax_3.set_title('MAD-aligned')
ax_3.set_xlabel('x\'')
ax_3.set_ylabel('y\'')
ax_3.set_box_aspect(1)
ax_3.set_aspect('equal')
pp.setp(ax_3.get_xticklabels(), rotation = 30, ha = 'right', rotation_mode = 'anchor')
pp.setp(ax_3.get_yticklabels(), rotation = 60, ha = 'right', rotation_mode = 'anchor')
plot_3 = ax_3.plot(
        mad_rotated[:, 0], mad_rotated[:, 1],
        marker = '.', markersize = 1,
        linestyle = '',
        color = 'red'
        )

pp.show() """



' ========== mirai dataset ========== '


SAMPLE = False
RATIO = 0.1


# - load -

df = pd.read_csv('../datasets/mirai.csv')
df = df[df['attack_flag'] == 0]

orderless = [
        'src_ip_addr',
        'src_port',
        'dst_ip_addr',
        'dst_port',
        'protocol',
        'flow_protocol',
        'attack_flag',
        'attack_step',
        'attack_name'
        ]
df.drop(orderless, axis = 'columns', inplace = True)
data = df.to_numpy(dtype = 'float64', copy = True)

if SAMPLE:
    index = np.arange(data.shape[0])
    index = np.random.choice(index, size = int(index.shape[0] * RATIO), replace = False)
    data = data[index]


# - aligned -

#standard deviation
pca = PCA(svd_solver = 'full')
pca.fit(data)
std_aligned = pca.transform(data)

#median absolute deviation
spread = dispersion(data)
eig, eigmat = la.eigh(spread)
eig = np.flip(eig, axis = 0)
eigmat = np.flip(eigmat, axis = 1)
mad_aligned = data @ eigmat


# - analysis -

#minmax
std_minmax = std_aligned.max(axis = 0) - std_aligned.min(axis = 0)
mad_minmax = mad_aligned.max(axis = 0) - mad_aligned.min(axis = 0)

#std
std_std = std_aligned.std(axis = 0)
mad_std = mad_aligned.std(axis = 0)

#mad
std_mad = mad(std_aligned, axis = 0)
mad_mad = mad(mad_aligned, axis = 0)


# - estimation -

estimator = BareEstimator()
width = 10000

# How can the alignment with respect to standard deviation more stable than median absolute deviation?
# It seems the points are not aligned properly, because it is not continuous-PSD.
# The 2-dimensional test above is just a lucky case where all the eigenvalues are semipositive.
std_dim = estimator(std_aligned, width, exact = True)
mad_dim = estimator(mad_aligned, width, exact = True)

print('  >> width = {} <<'.format(width))
print('estimated (std-aligned): {}'.format(std_dim))
print('estimated (MAD-aligned): {}'.format(mad_dim))
