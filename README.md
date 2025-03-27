# dimension-estimation


#### Description

This is a high-dimensional extension of the previous work 'CubeDimAE'. The problem of the intrinsic dimension of high-dimensional datasets, MNIST for example, all being measured as zeros, has been mitigated by aligning the axes with the principal components to stabilize the connections. Whether the new results are correct is to be verified, and it seems to require building more advanced models.

Also the iterative scanning, where multiple grid widths are tested, is eliminated as redundant. Current way of determining the width is very sensitive to outliers, so quantile-based scaling is being considered.

In terms of optimization, there is much room for improvement. It involves a computation of two symmetric arrays, which by definition half the computation is useless. Devising a way of computing only the half without being affected by instantiation and indexing overhead.
