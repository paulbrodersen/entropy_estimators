#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+entropy_estimators@gmail.com>

# Author: Paul Brodersen <paulbrodersen+entropy_estimators@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import itertools

from scipy.spatial import cKDTree
from scipy.special import gamma, digamma
from scipy.stats import multivariate_normal

"""
TODO:
- write test for get_pid()
- get_pmi() with normalisation fails test
"""

log = np.log10 # i.e. information measures are in bits
# log = np.log # i.e. information measures are in nats

def convert2rank(arr):
    # return np.argsort(arr, axis=0) / np.float(arr.shape[0])
    return (arr - np.nanmin(arr, axis=0)[None,:]) / (np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0))

def det(array_or_scalar):
    if array_or_scalar.size > 1:
        return np.linalg.det(array_or_scalar)
    else:
        return array_or_scalar

def get_h_mvn(X):

    """
    Computes the entropy of a multivariate Gaussian distribution:

        H(X) = (1/2) * log((2 * pi * e)^d * det(cov(X)))

    Arguments:
    ----------
        X: array of floats with dimensions (N samples, d data dimensions)
            samples from multivariate normal distribution;

    Returns:
    --------
        H: scalar
            entropy estimate
    """

    d = X.shape[1]
    H  = 0.5 * log((2 * np.pi * np.e)**d * det(np.cov(X.T)))
    return H

def get_mi_mvn(X, Y):
    """
    Computes the mutual information I between two multivariate normal random
    variables, X and Y:

        I(X, Y) = H(X) + H(Y) - H(X, Y)

    Arguments:
        X, Y: arrays of floats with dimensions (N samples, d data dimensions)
            samples from X, Y

    Returns:
        I: scalar
            mutual information between X and Y
    """

    d = X.shape[1]

    HX  = 0.5 * log((2 * np.pi * np.e)**d * det(np.cov(X.T)))
    HY  = 0.5 * log((2 * np.pi * np.e)**d * det(np.cov(Y.T)))
    HXY = 0.5 * log((2 * np.pi * np.e)**(2*d) * det(np.cov(X.T, y=Y.T)))

    return HX + HY - HXY

def get_pmi_mvn(X, Y, Z):
    """
    Computes the partial mutual information PMI between two multivariate normal random
    variables, X and Y, while conditioning on a third MVN RV, Z:

        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X, Y, Z) - H(Z)

    where:

        H(Z)     = (1/2) * log(det(2 * pi * e * cov(Z)))
        H(X,Z)   = (1/2) * log(det(2 * pi * e * cov(XZ)))
        H(Y,Z)   = (1/2) * log(det(2 * pi * e * cov(YZ)))
        H(X,Y,Z) = (1/2) * log(det(2 * pi * e * cov(XYZ)))

    Arguments:
        X, Y, Z: arrays of floats with dimensions (N samples, d data dimensions)
            samples from X, Y, Z

    Returns:
        pmi: scalar
            partial mutual information between X and Y conditioned on Z
    """

    d = X.shape[1]
    HZ   = 0.5 * log((2 * np.pi * np.e)**d * det(np.cov(Z.T)))
    HXZ  = 0.5 * log((2 * np.pi * np.e)**(2*d) * det(np.cov(X.T, y=Z.T)))
    HYZ  = 0.5 * log((2 * np.pi * np.e)**(2*d) * det(np.cov(Y.T, y=Z.T)))
    HXYZ = 0.5 * log((2 * np.pi * np.e)**(3*d) * det(np.cov(np.c_[X,Y,Z].T)))

    return HXZ + HYZ - HXYZ - HZ

def get_h(x, normalize=False, k=1, norm=np.inf, min_dist=0.):
    """
    Estimates the entropy H of a random variable x (in nats) based on
    the kth-nearest neighbour distances between point samples.

    @reference:
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9–16.

    Arguments:
    ----------
        x: (N observations, d data dimensions) array of floats
            the random variable
        k: int
            kth nearest neighbour to use in density estimate; imposes smoothness
            on the underlying probability distribution
        norm: 1, 2, or np.inf (default)
            p-norm used when computing k-nearest neighbour distances

    Returns:
    --------
        H: scalar
            entropy H(x)
    """

    N, d = x.shape

    if normalize:
        import warnings
        warnings.warn('Normalisation fundamentally breaks the KL-estimator! Only use option if you know what you are doing.')
        x = convert2rank(x)

    # volume of the d-dimensional unit ball...
    if norm == np.inf: # max norm:
        log_c_d = 0
    elif norm == 2: # euclidean norm
        log_c_d = (d/2.) * log(np.pi) -log(gamma(d/2. +1))
    elif norm == 1:
        pass
    else:
        raise NotImplementedError("Variable 'norm' either 1, 2 or np.inf")

    kdtree = cKDTree(x)

    # query all points -- k+1 as query point also in initial set
    distances, idx = kdtree.query(x, k + 1, eps=0, p=norm)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(log(2*distances)) # where did the 2 come from? radius -> diameter
    h = -digamma(k) + digamma(N) + log_c_d + (d / float(N)) * sum_log_dist

    return h

def get_mi(x, y, normalize=False, k=1, norm=np.inf, estimator='ksg'):
    """
    Estimates the mutual information (in nats) between two point clouds, x and y,
    in a D-dimensional space.

    I(X,Y) = H(X) + H(Y) - H(X,Y)

    @reference:
    Kraskov, Stoegbauer & Grassberger (2004). Estimating mutual information. PHYSICAL REVIEW E 69, 066138

    Arguments:
    ----------
        x, y:
            arrays of floats with dimensions N samples x D data dimensions
        normalize: bool (default=False)
            if True, data values are replaced by their rank in each dimension
            (destroys linear correlations)
        k:
            kth nearest neighbour to use in density estimate; imposes smoothness
            on the underlying probability distribution
        norm: 1, 2, or np.inf (default)
            p-norm used when computing k-nearest neighbour distances
        estimator: 'ksg' (default)or 'naive'
            'ksg': see Kraskov, Stoegbauer & Grassberger (2004) Estimating mutual information, eq(8).
            'naive': entropies are calculated individually using the Kozachenko-Leonenko estimator implemented in get_h()

    Returns:
    --------
        mi: scalar float
            mutual information
    """

    if normalize:
        x = convert2rank(x)
        y = convert2rank(y)

    # construct state array for the joint process:
    xy = np.c_[x,y]

    if estimator == 'naive':
        # compute individual entropies
        hx  = get_h(x,  normalize=False, k=k, norm=norm)
        hy  = get_h(y,  normalize=False, k=k, norm=norm)
        hxy = get_h(xy, normalize=False, k=k, norm=norm)

        # compute mi
        mi = hx + hy - hxy

    elif estimator == 'ksg':

        # store data pts in kd-trees for efficient nearest neighbour computations
        # TODO: choose a better leaf size
        x_tree  = cKDTree(x)
        y_tree  = cKDTree(y)
        xy_tree = cKDTree(xy)

        # kth nearest neighbour distances for every state
        # query with k=k+1 to return the nearest neighbour, not counting the data point itself
        dist, idx = xy_tree.query(xy, k=k+1, p=norm)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the x-subspace is strictly < epsilon
        # repeat for the y subspace
        N = len(x)
        nx = np.empty(N, dtype=np.int)
        ny = np.empty(N, dtype=np.int)
        for ii in xrange(N):
            nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=norm)) - 1
            ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=norm)) - 1

        mi = digamma(k) - np.mean(digamma(nx+1) + digamma(ny+1)) + digamma(N) # version (1)
        # mi = digamma(k) -1./k -np.mean(digamma(nx) + digamma(ny)) + digamma(N) # version (2)

    elif estimator == 'lnc':
        # TODO: (only if you can find some decent explanation on how to set alpha!)
        raise NotImplementedError("Estimator is one of 'naive', 'ksg'; currently: {}".format(estimator))

    else:
        raise NotImplementedError("Estimator is one of 'naive', 'ksg'; currently: {}".format(estimator))

    return mi

def get_pmi(x, y, z, normalize=False, k=1, norm=np.inf, estimator='fp'):
    """
    Estimates the partial mutual information (in nats), i.e. the
    information between two point clouds, x and y, in a D-dimensional
    space while conditioning on a third variable z.

    I(X,Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

    The estimators are based on:

    @reference:
    Frenzel & Pombe (2007) Partial mutual information for coupling analysis of multivariate time series
    Poczos & Schneider (2012) Nonparametric Estimation of Conditional Information and Divergences

    Arguments:
    ----------
        x, y, z:
            arrays of floats with dimensions N samples x D data dimensions
        normalize: bool (default=False)
            if True, data values are replaced by their rank in each dimension
            (destroys linear correlations)
        k:
            kth nearest neighbour to use in density estimate; imposes smoothness
            on the underlying probability distribution
        norm: 1, 2, or np.inf (default)
            p-norm used when computing k-nearest neighbour distances
        estimator: 'fp' (default), 'ps' or 'naive'
            'naive': entropies are calculated individually using the Kozachenko-Leonenko estimator implemented in get_h()
            'fp': Frenzel & Pombe estimator (effectively the KSG-estimator for mutual information)

    Returns:
    --------
        mi: scalar float
            mutual information

    """

    if normalize:
        x = convert2rank(x)
        y = convert2rank(y)
        z = convert2rank(z)

    # construct state array for the joint processes:
    xz  = np.c_[x,z]
    yz  = np.c_[y,z]
    xyz = np.c_[x,y,z]

    if estimator == 'naive':
        # compute individual entropies
        hz   = get_h(z,   k=k)
        hxz  = get_h(xz,  k=k)
        hyz  = get_h(yz,  k=k)
        hxyz = get_h(xyz, k=k)

        pmi =  hxz + hyz - hxyz - hz

    elif estimator == 'fp':

        # construct k-d trees
        z_tree   = cKDTree(z)
        xz_tree  = cKDTree(xz)
        yz_tree  = cKDTree(yz)
        xyz_tree = cKDTree(xyz)

        # kth nearest neighbour distances for every state
        # query with k=k+1 to return the nearest neighbour, not the data point itself
        dist, idx = xyz_tree.query(xyz, k=k+1, p=norm)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the relevant subspace is strictly < epsilon
        N, _ = x.shape
        nxz = np.empty(N, dtype=np.int)
        nyz = np.empty(N, dtype=np.int)
        nz  = np.empty(N, dtype=np.int)

        for ii in xrange(N):
            nz[ii]  = len( z_tree.query_ball_point( z_tree.data[ii], r=epsilon[ii], p=norm)) - 1
            nxz[ii] = len(xz_tree.query_ball_point(xz_tree.data[ii], r=epsilon[ii], p=norm)) - 1
            nyz[ii] = len(yz_tree.query_ball_point(yz_tree.data[ii], r=epsilon[ii], p=norm)) - 1

        pmi = digamma(k) + np.mean(digamma(nz +1) -digamma(nxz +1) -digamma(nyz +1))

    elif estimator == 'ps':
        # this is the correct implementation of the estimator,
        # but the estimators is crap

        # construct k-d trees
        xz_tree  = cKDTree(xz,  leafsize=2*k)
        yz_tree  = cKDTree(yz,  leafsize=2*k)

        # determine k-nn distances
        rxz = np.empty(N, dtype=np.int)
        ryz = np.empty(N, dtype=np.int)

        rxz, dummy = xz_tree.query(xz, k=k+1, p=norm) # +1 to account for distance to itself
        ryz, dummy = yz_tree.query(xz, k=k+1, p=norm) # +1 to account for distance to itself; xz NOT a typo

        pmi = yz.shape[1] * np.mean(log(ryz[:,-1]) - log(rxz[:,-1])) # + log(N) -log(N-1) -1.

    else:
        raise NotImplementedError("Estimator one of 'naive', 'fp', 'ps'; currently: {}".format(estimator))

    return pmi

def get_imin(x1, x2, y, normalize=False, k=1, norm=np.inf):
    """
    Estimates the average specific information (in nats) between a random variable Y
    and two explanatory variables, X1 and X2.

    I_min(Y; X1, X2) = \sum_{y \in Y} p(y) min_{X \in {X1, X2}} I_spec(y; X)

    where

    I_spec(y; X) = \sum_{x \in X} p(x|y) \log(p(y|x) / p(x))

    @reference:
    Williams & Beer (2010). Nonnegative Decomposition of Multivariate Information. arXiv:1004.2515v1
    Kraskov, Stoegbauer & Grassberger (2004). Estimating mutual information. PHYSICAL REVIEW E 69, 066138

    Arguments:
    ----------
        x1, x2, y:
            arrays of floats with dimensions N samples x D data dimensions
        normalize: bool (default=False)
            if True, data values are replaced by their rank in each dimension
            (destroys linear correlations)
        k:
            kth nearest neighbour to use in density estimate; imposes smoothness
            on the underlying probability distribution
        norm: 1, 2, or np.inf (default)
            p-norm used when computing k-nearest neighbour distances

    Returns:
    --------
        i_min: scalar float

    """

    if normalize:
        y = convert2rank(y)

    y_tree  = cKDTree(y)

    N = len(y)
    i_spec = np.zeros((2, N))

    for jj, x in enumerate([x1, x2]):

        if normalize:
            x = convert2rank(x)

        # construct state array for the joint processes:
        xy = np.c_[x,y]

        # store data pts in kd-trees for efficient nearest neighbour computations
        # TODO: choose a better leaf size
        x_tree  = cKDTree(x)
        xy_tree = cKDTree(xy)

        # kth nearest neighbour distances for every state
        # query with k=k+1 to return the nearest neighbour, not counting the data point itself
        dist, idx = xy_tree.query(xy, k=k+1, p=norm)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the x-subspace is strictly < epsilon
        # repeat for the y subspace
        nx = np.empty(N, dtype=np.int)
        ny = np.empty(N, dtype=np.int)
        for ii in xrange(N):
            nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=norm)) - 1
            ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=norm)) - 1

        i_spec[jj] = digamma(k) - digamma(nx+1) + digamma(ny+1) + digamma(N) # version (1)

    i_min = np.mean(np.min(i_spec, 0))

    return i_min

def get_pid(x1, x2, y, normalize=False, k=5, norm=np.inf):

    """
    Estimates the partial information decomposition (in nats) between a random variable Y
    and two explanatory variables, X1 and X2.

    I(X1, X2; Y) = synergy + unique_{X1} + unique_{X2} + redundancy

    redundancy = I_{min}(X1, X2; Y)
    unique_{X1} = I(X1; Y) - redundancy
    unique_{X2} = I(X2; Y) - redundancy
    synergy = I(X1, X2; Y) - I(X1; Y) - I(X2; Y) + redundancy

    The estimator is based on:

    @reference:
    Williams & Beer (2010). Nonnegative Decomposition of Multivariate Information. arXiv:1004.2515v1
    Kraskov, Stoegbauer & Grassberger (2004). Estimating mutual information. PHYSICAL REVIEW E 69, 066138

    For a critique of I_min as a redundancy measure, see
    Bertschinger et al. (2012). Shared Information – New Insights and Problems in Decomposing Information in Complex Systems. arXiv:1210.5902v1
    Griffith & Koch (2014). Quantifying synergistic mutual information. arXiv:1205.4265v6

    Arguments:
    ----------
        x1, x2, y:
            arrays of floats with dimensions N samples x D data dimensions
        normalize: bool (default=False)
            if True, data values are replaced by their rank in each dimension
            (destroys linear correlations)
        k:
            kth nearest neighbour to use in density estimate; imposes smoothness
            on the underlying probability distribution
        norm: 1, 2, or np.inf (default)
            p-norm used when computing k-nearest neighbour distances

    Returns:
    --------
        synergy: scalar float
            information about Y encoded by the joint state of x1 and x2
        unique_x1: scalar float
            information about Y encoded uniquely by x1
        unique_x2: scalar float
            information about Y encoded uniquely by x2
        redundancy: scalar float
            information about Y encoded by either x1 or x2

    """

    mi_x1y = get_mi(x1, y, normalize=normalize, k=k, norm=norm)
    mi_x2y = get_mi(x2, y, normalize=normalize, k=k, norm=norm)
    mi_x1x2y = get_mi(np.c_[x1, x2], y, normalize=normalize, k=k, norm=norm)

    redundancy = get_imin(x1, x2, y, normalize=normalize, k=k, norm=norm)

    unique_x1 = mi_x1y - redundancy
    unique_x2 = mi_x2y - redundancy
    synergy = mi_x1x2y - mi_x1y - mi_x2y + redundancy

    return synergy, unique_x1, unique_x2, redundancy

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

def get_mvn_data(total_rvs, dimensionality=2, scale_sigma_offdiagonal_by=1., total_samples=1000):
    data_space_size = total_rvs * dimensionality

    # initialise distribution
    mu = np.random.randn(data_space_size)
    sigma = np.random.rand(data_space_size, data_space_size)
    # sigma = 1. + 0.5*np.random.randn(data_space_size, data_space_size)

    # ensures that sigma is positive semi-definite
    sigma = np.dot(sigma.transpose(), sigma)

    # scale off-diagonal entries -- might want to change that to block diagonal entries
    # diag = np.diag(sigma).copy()
    # sigma *= scale_sigma_offdiagonal_by
    # sigma[np.diag_indices(len(diag))] = diag

    # scale off-block diagonal entries
    d = dimensionality
    for ii, jj in itertools.product(range(total_rvs), repeat=2):
        if ii != jj:
            sigma[d*ii:d*(ii+1), d*jj:d*(jj+1)] *= scale_sigma_offdiagonal_by

    # get samples
    samples = multivariate_normal(mu, sigma).rvs(total_samples)

    return [samples[:,ii*d:(ii+1)*d] for ii in range(total_rvs)]


def test_get_h(normalize=False, k=5, norm=np.inf):
    X, = get_mvn_data(total_rvs=1,
                      dimensionality=2,
                      scale_sigma_offdiagonal_by=1.,
                      total_samples=1000)

    analytic = get_h_mvn(X)
    kozachenko = get_h(X, normalize, k, norm)

    print "analytic result: {: .5f}".format(analytic)
    print "K-L estimator: {: .5f}".format(kozachenko)

    return

def test_get_mi(normalize=False, k=5, norm=np.inf):

    X, Y = get_mvn_data(total_rvs=2,
                        dimensionality=2,
                        scale_sigma_offdiagonal_by=1., # 0.1, 0.
                        total_samples=10000)

    # solutions
    analytic = get_mi_mvn(X, Y)
    naive = get_mi(X, Y, normalize=normalize, k=k, norm=norm, estimator='naive')
    ksg   = get_mi(X, Y, normalize=normalize, k=k, norm=norm, estimator='ksg')

    print "analytic result: {: .5f}".format(analytic)
    print "naive estimator: {: .5f}".format(naive)
    print "KSG estimator:   {: .5f}".format(ksg)
    print

    # for automated testing:
    assert np.isclose(analytic, naive, rtol=0.5, atol=0.5), "Naive MI estimate strongly differs from expectation!"
    assert np.isclose(analytic, ksg, rtol=0.5, atol=0.5), "KSG MI estimate strongly differs from expectation!"

    return

def test_get_pmi(normalize=False, k=5, norm=np.inf):

    X, Y, Z = get_mvn_data(total_rvs=3,
                        dimensionality=2,
                        scale_sigma_offdiagonal_by=1.,
                        total_samples=10000)

    # solutions
    analytic = get_pmi_mvn(X, Y, Z)
    naive = get_pmi(X, Y, Z, normalize=normalize, k=k, norm=norm, estimator='naive')
    fp    = get_pmi(X, Y, Z, normalize=normalize, k=k, norm=norm, estimator='fp')

    print "analytic result: {: .5f}".format(analytic)
    print "naive estimator: {: .5f}".format(naive)
    print "FP estimator:   {: .5f}".format(fp)
    print

    # for automated testing:
    assert np.isclose(analytic, naive, rtol=0.5, atol=0.5), "Naive MI estimate strongly differs from expectation!"
    assert np.isclose(analytic, fp,    rtol=0.5, atol=0.5), "FP MI estimate strongly differs from expectation!"

    return

def test_get_pid(normalize=False, k=5, norm=np.inf):
    # rdn -> only redundant information

    # unq -> only unique information

    # xor -> only synergistic information

    return
