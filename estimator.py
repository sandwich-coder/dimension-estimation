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
            X: np.ndarray,
            batch_count = None,
            exact: bool = False,
            fast: bool = False,
            divisions: int = 10
            ):
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The shape must be the dataset standard.')
        if batch_count is not None:
            if not isinstance(batch_count, int):
                raise TypeError('\'batch_count\' should be an integer.')
            if batch_count < 1:
                raise ValueError('\'batch_count\' must be greater than 1.')
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

        #measured
        max_range = np.max(
                X.max(axis = 0) - X.min(axis = 0),
                axis = 0
                )

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
        width = max_range / np.float64(divisions)
        if divisions % 2 == 0:
            tile = X / width - np.float64(0.5)
        else:
            tile = X / width
        tile = tile.round().astype(tile_dtype)
        tile = np.unique(tile, axis = 0)


        # - counted -

        if batch_count is not None:

            batch = copy(np.array_split(tile, batch_count, axis = 0))
            if batch_count > tile.shape[0]:
                effective_length = tile.shape[0] % batch_count
                batch = batch[:effective_length]

            adjacency = []
            for llll in tqdm(batch, colour = 'magenta'):

                _ = llll.reshape([llll.shape[0], 1, llll.shape[1]])
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

        else:

            compared = tile.reshape([tile.shape[0], 1, tile.shape[1]]).copy()
            compared = compared.repeat(tile.shape[0], axis = 1)

            all_ = compared.swapaxes(0, 1).copy()

            distance = np.max(
                    np.absolute(compared - all_),
                    axis = 2
                    )
            is_adjacent = distance == 1

            adjacency = is_adjacent.astype('int64').sum(axis = 1, dtype = 'int64')



        dimension = np.log(adjacency.mean() + 1, dtype = 'float64') / np.log(3, dtype = 'float64')
        if exact:
            dimension = dimension.tolist()
        else:
            dimension = dimension.round().astype('int64')
            dimension = dimension.tolist()
        return dimension
