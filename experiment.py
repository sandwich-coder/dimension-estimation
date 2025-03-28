from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np
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

cifar = images.CIFAR10(
        root = 'datasets',
        train = True,
        download = True,
        transform = ToTensor()
        ).data
cifar = cifar.reshape([cifar.shape[0], -1])

#select
data = cifar.copy()


# - processed -

data = data.astype('float64')

scaler = MinMaxScaler(feature_range = (-1, 1))
scaler.fit(data)
data = scaler.transform(data)


estimator = DimensionEstimator()
