from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np
from scipy import linalg as la
from scipy.fft import fftn as ft, ifftn as ift, fftfreq, fftshift
from scipy.spatial.distance import pdist, cdist
import torch

from estimator import DimensionEstimator

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets as images
from torchvision.transforms import ToTensor


# - load -

digits = images.MNIST(
        root = 'datasets',
        train = True,
        download = True,
        transform = ToTensor()
        ).data.numpy()
digits = digits.reshape([digits.shape[0], -1])

cloths = images.FashionMNIST(
        root = 'datasets',
        train = True,
        download = True,
        transform = ToTensor()
        ).data.numpy()
cloths = cloths.reshape([cloths.shape[0], -1])

cifar10 = images.CIFAR10(
        root = 'datasets',
        train = True,
        download = True,
        transform = ToTensor()
        ).data
cifar10 = cifar10.reshape([cifar10.shape[0], -1])

cifar100 = images.CIFAR100(
        root = 'datasets',
        train = True,
        download = True,
        transform = ToTensor()
        ).data
cifar100 = cifar100.reshape([cifar100.shape[0], -1])

#select
data = digits.copy()


# - processed -

data = data.astype('float64')

scaler = MinMaxScaler(feature_range = (-1, 1))    # 'MinMaxScaler' scales features individually.
scaler.fit(data)
data = scaler.transform(data)


estimator = DimensionEstimator()
dimension = estimator(data, exact = True, truncate = True)
print('dimension: {}'.format(dimension))
