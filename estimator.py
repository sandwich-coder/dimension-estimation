from copy import deepcopy as copy
import os, sys
import types
import logging
import numpy as np

from sklearn.decomposition import PCA
from tqdm import tqdm


class DimensionEstimator:
    def __repr__(self):
        return 'dimension estimator'

    def __call__(
            self,
            X,
            batch_count = 1000,
            exact = False,
            fast = False,
            divisions = 10
            ):
        if type(X) != np.ndarray:
            raise TypeError('The input must be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The shape must be the dataset standard.')
        if not isinstance(batch_count, int):
            raise TypeError('\'batch_count\' should be an integer.')
        if batch_count < 1:
            raise ValueError('\'batch_count\' must be positive.')
        if not isinstance(exact, bool):
            raise TypeError('\'exact\' should be boolean.')
        if not isinstance(fast, bool):
            raise TypeError('\'fast\' should be boolean.')
        if not isinstance(divisions, int):
            raise TypeError('\'divisions\' should be an integer.')
        if divisions < 2:
            raise ValueError('\'divisions\' must be greater than 1.')
        if divisions > 10000:
            raise ValueError('\'divisions\' greater than 10000 is not supported.')
        if fast:
            retained_variance = 0.9
            logging.warning('The dataset is truncated at 90% total variance for faster computation. It may overtruncate for those of few features.')
        else:
            retained_variance = None
        if divisions < 50:
            tile_dtype = np.int8
        else:
            tile_dtype = np.int16

        #oriented
        pca = PCA(n_components = retained_variance, svd_solver = 'full')
        pca.fit(X)
        X = pca.transform(X)

        # A tile is always adjacent to all the others in the case of two divisions.
        if divisions == 2:
            binary = np.where(X >= 0, 1, -1)
            binary = binary.astype('int8')

            occupied = np.unique(binary, axis = 0)
            dimension = np.log(occupied.shape[0], dtype = 'float64') / np.log(2, dtype = 'float64')
            if exact:
                dimension = dimension.tolist()
            else:
                dimension = dimension.round().astype('int64')
                dimension = dimension.tolist()
            return dimension


        #quantized
        max_range = np.max(X.max(axis = 0) - X.min(axis = 0), axis = 0)
        width = max_range / np.float64(divisions)
        if divisions % 2 != 0:
            tile = X / width
        else:
            tile = X / width - np.float64(0.5)
        tile = tile.round().astype(tile_dtype)
        tile = np.unique(tile, axis = 0)


        # - counted -

        batch = copy(np.array_split(tile, batch_count, axis = 0))
        if batch_count > tile.shape[0]:
            effective_length = tile.shape[0] % batch_count
            batch = batch[:effective_length]

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


        dimension = np.log(adjacency.mean(axis = 0) + 1, dtype = 'float64') / np.log(3, dtype = 'float64')
        if exact:
            dimension = dimension.tolist()
        else:
            dimension = dimension.round().astype('int64')
            dimension = dimension.tolist()
        return dimension
