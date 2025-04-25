from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np
from scipy import linalg as la
from scipy.fft import fftn, ifftn, fftfreq, fftshift
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation as mad1d
import matplotlib as mpl
from matplotlib import pyplot as pp

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll, make_s_curve
from tqdm import tqdm
from torchvision import datasets as images
from torchvision.transforms import ToTensor


#spatial median
#slow and optional
def median(X):
    if type(X) != np.ndarray:
        raise TypeError('Input must be a \'numpy.ndarray\'.')
    if X.dtype != np.float64:
        X = X.astype('float64')
    if X.ndim != 2:
        raise ValueError('The shape must be the dataset standard.')
    
    def func(y):
        y = y.reshape([1, y.shape[0]])
        distance = cdist(y, X)
        return distance.sum(dtype = 'float64')
    
    initial = X.mean(axis = 0, dtype = 'float64')
    result = minimize(func, initial, method = 'Nelder-Mead')
    
    return result.x
    

# This method builds on the fact that
# for continuous semipositive distribution,
# its median squared equals the median of it squared.
def mad_align(X):
    if type(X) != np.ndarray:
        raise TypeError('Input must be a \'numpy.ndarray\'.')
    if X.dtype != np.float64:
        X = X.astype('float64')
    if X.ndim != 2:
        raise ValueError('The shape must be the dataset standard.')
    
    center = median(X)
    
    mad = np.median(np.absolute(
        X - center.reshape([1, center.shape[0]])
        ), axis = 0)
    
    _ = mad.reshape([1, mad.shape[0]])
    comad = _.transpose() @ _
    eig, eigmat = la.eigh(comad)
    
    centered = X - center.reshape([1, center.shape[0]])
    rotated = centered @ np.flip(eigmat, axis = 1)
    return rotated
    



' ========== 2-dimensional test ========== '


x1 = np.arange(10, dtype = 'float64')
data = np.stack([x1, x1], axis = 1)

outlier = np.array([
    [10, -100]
    ], dtype = 'float64')
data = np.concatenate([data, outlier], axis = 0)


# - aligned -

pca = PCA(svd_solver = 'full')
pca.fit(data)
pca_aligned = pca.transform(data)

mad_aligned = mad_align(data)


# - plot -

fig = pp.figure(layout = 'constrained', figsize = (12, 5))
fig.suptitle('Comparison')
gs = fig.add_gridspec(nrows = 1, ncols = 2, hspace = 0.1)

ax_1 = fig.add_subplot(gs[1 - 1])
ax_1.set_title('PCA')
ax_1.set_box_aspect(1)
ax_1.set_aspect('equal')
ax_1.set_xlabel(r'$x_{1}$')
ax_1.set_ylabel(r'$x_{2}$')
plot_1 = ax_1.plot(
    pca_aligned[:, 0], pca_aligned[:, 1],
    marker = 'o', markersize = 3,
    linestyle = '',
    color = 'blue'
    )

ax_2 = fig.add_subplot(gs[2 - 1])
ax_2.set_title('MAD')
ax_2.set_box_aspect(1)
ax_2.set_aspect('equal')
ax_2.set_xlabel(r'$x_{1}$')
ax_2.set_ylabel(r'$x_{2}$')
plot_2 = ax_2.plot(
    mad_aligned[:, 0], mad_aligned[:, 1],
    marker = 'o', markersize = 3,
    linestyle = '',
    color = 'red'
    )



' ========== datasets ========== ''''


#swiss-roll
roll = make_swiss_roll(n_samples = 1000)[0]

#mirai
mirai = pd.read_csv('../datasets/mirai.csv')
mirai = mirai[mirai['attack_flag'] == 0]
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
mirai.drop(orderless, axis = 'columns', inplace = True)
mirai = mirai.to_numpy(dtype = 'float64')


#digits

digit = images.MNIST(
    root = '../datasets',
    train = True,
    transform = ToTensor(),
    download = True
    ).data.numpy()
digit = digit.reshape([digit.shape[0], -1])

pca = PCA(n_components = 0.9, svd_solver = 'full')
pca.fit(digit)
digit = pca.transform(digit)


#select
data = digit.copy()


# - rotated -

pca = PCA(svd_solver = 'full')
pca.fit(data)
pca_aligned = pca.transform(data)

mad_aligned = mad_align(data)


#minmax
pca_minmax = pca_aligned.max(axis = 0) - pca_aligned.min(axis = 0)
mad_minmax = mad_aligned.max(axis = 0) - mad_aligned.min(axis = 0)

#std
pca_std = pca_aligned.std(axis = 0)
mad_std = mad_aligned.std(axis = 0)

#mad
pca_mad = mad1d(pca_aligned, axis = 0)
mad_mad = mad1d(mad_aligned, axis = 0)


# - tile counting -

batch_count = 1000
divisions = 10

#select
input_ = mad_aligned

"""
range_ = np.max(
    input_.max(axis = 0) - input_.min(axis = 0),
    axis = 0)
width = range_ / np.float64(divisions)
"""
width = np.float64(350)

tile = input_ / width
tile = tile.round().astype('int64')
tile = np.unique(tile, axis = 0)

batch = copy(np.array_split(tile, batch_count, axis = 0))
if  tile.shape[0] < batch_count:
    batch = batch[:tile.shape[0]]

adjacency = []
for l in tqdm(batch, colour = 'magenta'):
    
    distance = cdist(l, tile, metric = 'chebyshev')
    is_adjacent = np.isclose(distance, 1)
    
    adjacency_batch = is_adjacent.astype('int64')
    adjacency_batch = adjacency_batch.sum(axis = 1, dtype = 'int64')
    adjacency.append(adjacency_batch)
    
adjacency = np.concatenate(adjacency, axis = 0)

dimension = np.log(adjacency.mean(axis = 0) + 1, dtype = 'float64') / np.log(3, dtype = 'float64')
print('\n')
print('dimension: {}'.format(dimension))'''
