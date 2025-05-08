from copy import deepcopy as copy
import logging
import numpy as np
from scipy.spatial.distance import pdist, cdist

from sklearn.ensemble import IsolationForest
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
            trim = False,
            truncate = False,
            divisions = 10
            ):
        if type(X) != np.ndarray:
            raise TypeError('Input must be a \'numpy.ndarray\'.')
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
        if not isinstance(trim, bool):
            raise TypeError('\'trim\' should be boolean.')
        if not isinstance(truncate, bool):
            raise TypeError('\'truncate\' should be boolean.')
        if not isinstance(divisions, int):
            raise TypeError('\'divisions\' should be an integer.')
        if divisions < 2:
            raise ValueError('\'divisions\' must be greater than 1.')
        if divisions > 10000:
            raise ValueError('\'divisions\' greater than 10000 is not supported.')
        if truncate:
            retained_variance = 0.9
            logging.warning('The dataset is truncated at 90% total variance for faster computation. It may overtruncate those of few features.')
        else:
            retained_variance = None
        
        #trimmed
        if trim:
            logging.info('The dataset is trimmed by the isolation forest.')
            forest = IsolationForest()
            forest.fit(X)
            valid = forest.predict(X)
            valid = np.where(valid == 1, True, False)
            X = X[valid]
        
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
        width = (X[:, 0].max(axis = 0) - X[:, 0].min(axis = 0)) / np.float64(divisions)
        if divisions % 2 != 0:
            tile = X / width
        else:
            tile = X / width - np.float64(0.5)
        tile = tile.round().astype('int64')
        tile = np.unique(tile, axis = 0)
        
        
        # - counted -

        batch = copy(np.array_split(tile, batch_count, axis = 0))
        if batch_count > tile.shape[0]:
            batch = batch[:tile.shape[0]]
        
        adjacency = []
        for l in tqdm(batch, colour = 'magenta'):
            
            distance = cdist(l, tile, metric = 'chebyshev')
            is_adjacent = np.isclose(distance, 1, atol = 0)
            
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
