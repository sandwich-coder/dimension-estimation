from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np
from scipy import linalg as la
from scipy.fft import fftn, ifftn, fftfreq, fftshift
from scipy.spatial.distance import pdist, cdist
import matplotlib as mpl
from matplotlib import pyplot as pp

import pandas as pd
from sklearn.decomposition import PCA

from mad_align_2d import dispersion


#load

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
data = df.to_numpy(dtype = 'float64')
