# Dimension Estimation


#### Description

This is a high-dimensional extension of the previous work 'CubeDimAE'. The problem of the intrinsic dimensions of high-dimensional datasets, MNIST for example, all being measured as zeros, has been mitigated by aligning the axes with the principal components to stabilize the connections. Whether the new results are correct is to be verified, and it seems to require building more advanced autoencoders.

There were some revisions to the method. First, the iterative scanning has been eliminated as it seemed redundant. It became a parameter 'divisions' instead, with the default 10. Second, the weighted averaging of adjacencies has been replaced by simple mean, on the basis that the sparsity of high-dimensional datasets makes the density meaningless.

There is much room for improvement. It involves a computation within a symmetric array, which by definition half the computation is useless. Devising a way of reducing without being affected by instantiation and indexing overhead.


## Example Usage

```python
import numpy as np
from sklearn.datasets import make_swiss_roll
from estimator import DimensionEstimator

data = make_swiss_roll(n_samples = 1000)[0]
estimator = DimensionEstimator()

dimension = estimator(data)
```
