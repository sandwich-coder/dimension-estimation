from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np
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


# Spatial median is set as the center of rotation.
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
