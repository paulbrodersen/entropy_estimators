# Entropy estimators

This module implements estimators for the entropy and other
information theoretic quantities of continuous distributions, including:

* entropy / Shannon information
* mutual information
* partial mutual information (and hence transfer entropy)
* specific information
* partial information decomposition

The estimators derive from the Kozachenko and Leonenko (1987)
estimator, which uses k-nearest neighbour distances to compute the
entropy of distributions, and extension thereof developed by Kraskov
et al (2004), and Frenzel and Pombe (2007).

Pendants for discrete variables will be added at a later date.

## Table of Contents

- [Installation](#installation)
- [Examples](#examples)
- [Functions](#functions)

## Installation

Easiest via pip:

``` shell
pip install entropy_estimators
```

## Examples

```python

import numpy as np
from entropy_estimators import continuous

# create some normal test data
X = np.random.randn(10000, 2)

# compute the entropy from the determinant of the multivariate normal distribution:
analytic = continuous.get_h_mvn(X)

# compute the entropy using the k-nearest neighbour approach
# developed by Kozachenko and Leonenko (1987):
kozachenko = continuous.get_h(X, k=5)

print(f"analytic result: {analytic:.5f}")
print(f"K-L estimator: {kozachenko:.5f}")

```
## Functions
### `get_h_mvn(x)`
Computes the entropy of a multivariate Gaussian distribution.

### `get_mi_mvn(x, y)`
Computes the mutual information between two multivariate normal random variables.

### `get_pmi_mvn(x, y, z)`
Computes the partial mutual information between two multivariate normal random variables while conditioning on a third variable.

### `get_h(x, k=1, norm='max', min_dist=0., workers=1)`
Estimates the entropy of a random variable based on k-nearest neighbor distances between point samples.

### `get_mi(x, y, k=1, normalize=None, norm='max', estimator='ksg', workers=1)`
Estimates the mutual information between two point clouds in a D-dimensional space.

### `get_pmi(x, y, z, k=1, normalize=None, norm='max', estimator='fp', workers=1)`
Estimates the partial mutual information between two variables while conditioning on a third variable.

### `get_imin(x1, x2, y, k=1, normalize=None, norm='max', workers=1)`
Estimates the average specific information between a random variable and two explanatory variables.

### `get_pid(x1, x2, y, k=1, normalize=None, norm='max')`
Estimates the partial information decomposition between a random variable and two explanatory variables.

