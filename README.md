# Dimension Estimation


#### Description

This is a high-dimensional extension of the previous work 'CubeDimAE'. The problem of the intrinsic dimensions of high-dimensional datasets, MNIST for example, all being measured as zeros, has been mitigated by aligning the axes with the principal components to stabilize the connections. Whether the new results are correct is to be verified, and it seems to require building more advanced autoencoders.

There were some revisions. First, the iterative scanning has been eliminated as it seemed redundant. It became a parameter 'divisions' instead, with the default 10. Second, the weighted averaging of adjacencies has been replaced by simple mean, on the basis that the sparsity of high-dimensional datasets makes the density meaningless.

Aside from some computation redundancies that should be eliminated, this algorithm relies on classical statistics, which believes real-world datasets generally behave as what the central limit theorem says, in how the axes are aligned. Albeit very successful, so many exceptions are encountered that "robust statistics", a major field of modern statistics, has emerged. To handle those exceptions and become a more reliable method of dimension estimation, many parts should switch from mean-variance to the median-mad paradigm.


## Use

#### tested environments
Ubuntu-24.04

#### WSL
```
git clone https://github.com/sandwich-coder/dimension-estimation
cd dimension-estimation
pip install --upgrade pip
pip install --upgrade --requirement requirement.txt
```

#### Example Usage
```python
from estimator import DimensionEstimator

import numpy as np
from sklearn.datasets import make_swiss_roll

data = make_swiss_roll(n_samples = 1000)[0]
estimator = DimensionEstimator()

dimension = estimator(data)
```
