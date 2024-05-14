> [!CAUTION]
> This package implements the KL-estimator and the KSG-estimator for the entropy and mutual information of continuous variables (and a few variants thereof).
> These estimators have issues, and for most purposes, should likely no longer be the first choice.
> Please read see [this comment](https://github.com/paulbrodersen/entropy_estimators/issues/11#issuecomment-2109577671) for a better explanation than I could give, as well as a link to an alternative implementation.

# Entropy estimators

This module implements estimators for the entropy and other
information theoretic quantities of continuous distributions, including:

* entropy / Shannon information (`get_h`),
* mutual information (`get_mi`),
* partial mutual information & transfer entropy (`get_pmi`),
* specific information (`get_imin`), and
* partial information decomposition (`get_pid`).

The estimators derive from the
[Kozachenko and Leonenko (1987)](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=ppi&paperid=797&option_lang=eng)
estimator, which uses k-nearest neighbour distances to compute the entropy of distributions, and extension thereof developed by
[Kraskov et al. (2004)](https://arxiv.org/abs/cond-mat/0305641),
and
[Frenzel and Pombe (2007)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.99.204101).

For **multivariate normal distributions**, the following quantities can be computed analytically from the covariance matrix.

* entropy (`get_h_mvn`),
* mutual information (`get_mi_mvn`), and
* partial mutual information & transfer entropy (`get_pmi_mvn`).


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

## Frequently asked questions

#### Why is the estimate of the mutual information negative? Shouldn't it always be positive?

Mutual information is a strictly positive quantity. However, its *estimate* need not be, and in fact, the nearest neighbour estimators are known to be biased estimators (Kraskov et al. 2004). Unfortunately, the bias appears to depend on multiple factors, primarily the number of samples and the choice of the `k` parameter, and thus cannot be known *a priori*. However, the bias itself can be estimated using a straightforward permutation / bootstrap approach:

1. Compute the mutual information estimate between two variables, X and Y.
2. Permute either variable (or both), and re-compute the estimate. The mutual information between randomised variables is zero, so this estimate represents the bias.
3. Repeat the previous step many times to obtain a robust estimate of the bias.


``` python
import numpy as np

from scipy.stats import multivariate_normal
from entropy_estimators import continuous

# create two variables with a mutual information that can be computed analytically
means = [0, 1]
covariance = np.array([[1, 0.5], [0.5, 1]])

def get_entropy(covariance):
    """Compute the entropy of multivariate normal distribution from the covariance matrix."""
    if np.size(covariance) > 1:
        dim = covariance.shape[0]
        det = np.linalg.det(covariance)
    else: # scalar
        dim = 1
        det = covariance
    return 0.5 * np.log((2 * np.pi * np.e)**dim * det)

hx  = get_entropy(covariance[0, 0])
hy  = get_entropy(covariance[1, 1])
hxy = get_entropy(covariance)
analytic_result = hx + hy - hxy

# compute the mutual information from samples using the KSG estimator
distribution = multivariate_normal(means, covariance)
X, Y = distribution.rvs(1000).T

k = 5
ksg_estimate = continuous.get_mi(X, Y, k=k)

print(f"Analytic result: {analytic_result:.3f} nats")
print(f"KSG estimate: {ksg_estimate:.3f} nats")
print(f"Difference: {analytic_result - ksg_estimate:.3f} nats")
# Analytic result: 0.144
# KSG estimate: 0.113 nats
# Difference: 0.031 nats

# bootstrap to determine the bias
total_repeats = 100
bias = 0
Y_shuffled = Y.copy()
for ii in range(total_repeats):
    np.random.shuffle(Y_shuffled) # shuffling occurs in-place!
    bias += continuous.get_mi(X, Y_shuffled, k=k)
bias /= total_repeats

print("--------------------------------------------------------------------------------")
print(f"Bias estimat: {bias:.3f} nats")
print(f"Corrected KSG estimate: {ksg_estimate - bias:.3f}")
print(f"Difference to analytic result: {analytic_result - (ksg_estimate - bias):.3f} nats")
# Bias estimat: -0.020 nats
# Corrected KSG estimate: 0.132
# Difference to analytic result: 0.012 nats
```

## Alternative Implementations

### Scipy

[`scipy.stats.entropy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) : entropy of a categorical variable

### Scikit-learn

 * [`sklearn.metrics.mutual_info_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html#sklearn.metrics.mutual_info_score) : mutual information between two categorical variables

 * [`skelarn.metrics.mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) :
 mutual information between two continuous variables; note that their implementation does not report negative mutual information scores and thus makes it impossible to compute bias corrections using the bootstrap approach outlined above.

### Non-parametric Entropy Estimation Toolbox (NPEET)

Alternative python implementations of the nearest-neighbour estimators for the entropy of continuous variables, the mutual information and the partial/conditioned mutual information ([link](https://github.com/gregversteeg/NPEET)). In principle, there are no major differences between their implementation and this repository. However, for large samples, their implementation may run a little slower as it uses lists as the primary data structure and doesn't support parallelisation. The implementation in this repository mostly uses numpy arrays, which allows vectorization of many calculations, and supports running operations on multiple cores by setting the `workers` argument to valus larger than one.
