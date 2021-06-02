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
