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

## Examples

```python

import numpy
from entropy_estimators import continuous

# create some multivariate normal test data
X, = continous.get_mvn_data(total_rvs=1,        # number of random variables
                            dimensionality=2,   # dimensionality of each RV
                            total_samples=1000) # samples

# compute the entropy from the determinant of the multivariate normal distribution:
analytic = continuous.get_h_mvn(X)

# compute the entropy using the k-nearest neighbour approach
# developed by Kozachenko and Leonenko (1987):
kozachenko = continuous.get_h(X, k=5)

print "analytic result: {: .5f}".format(analytic)
print "K-L estimator: {: .5f}".format(kozachenko)

```
