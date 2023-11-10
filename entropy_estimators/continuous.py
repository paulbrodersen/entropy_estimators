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


import functools
import numpy as np

from scipy.spatial import KDTree
from scipy.special import gamma, digamma
from scipy.stats   import rankdata


log = np.log   # i.e. information measures are in nats
# log = np.log2  # i.e. information measures are in bits


def convert_vectors_to_2d_arrays_if_any(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for ii, arg in enumerate(args):
            if isinstance(arg, (list, tuple, np.ndarray)):
                if np.ndim(arg) == 1:
                    args[ii] = np.array(arg)[:, np.newaxis]
                elif np.ndim(arg) == 2:
                    pass
                else:
                    raise ValueError("Arrays should have one or two dimensions.")
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                if np.ndim(v) == 1:
                    kwargs[k] = v[:, np.newaxis]
                elif np.ndim(v) == 2:
                    pass
                else:
                    raise ValueError("Arrays should have one or two dimensions.")
        return func(*args, **kwargs)
    return wrapper


def unit_interval(arr):
    return (arr - np.nanmin(arr, axis=0)[None,:]) / (np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0))


def rank(arr):
    return np.apply_along_axis(rankdata, 0, arr)


def det(array_or_scalar):
    if array_or_scalar.size > 1:
        return np.linalg.det(array_or_scalar)
    else:
        return array_or_scalar


@convert_vectors_to_2d_arrays_if_any
def get_h_mvn(x, normalized=False):
    """
    Computes the entropy of a multivariate Gaussian distribution:

    H(X) = (1/2) * log((2 * pi * e)^d * det(cov(X)))

    Arguments:
    ----------
    x: (n, d) ndarray
        n samples from a d-dimensional multivariate normal distribution

    normalized: bool
        normalize distribution, if `True` each component is normalized such that
        its standard deviation is `1` and the covariance matrix becomes equal to
        the Pearson correlation coefficients and the entropy becomes invariant
        under linear transformation

    Returns:
    --------
    h: float
        entropy H(X)
    """

    d = x.shape[1]
    h  = 0.5 * log((2 * np.pi * np.e)**d * det(np.cov(x.T)))
    if d == 1:
        if normalized:
            h = 0.5 * np.log(2 * np.pi * np.e)
        else:
            h = 0.5 * np.log((2 * np.pi * np.e) * np.var(x, ddof=1))
    else:
        if normalized:
            h = 0.5 * np.log((2 * np.pi * np.e)**d * det(np.corrcoef(x.T)))
        else:
            h = 0.5 * np.log((2 * np.pi * np.e)**d * det(np.cov(x.T)))
    return h


@convert_vectors_to_2d_arrays_if_any
def get_mi_mvn(x, y):
    """
    Computes the mutual information I between two multivariate normal random
    variables, X and Y:

    I(X, Y) = H(X) + H(Y) - H(X, Y)

    Arguments:
    ----------
    x, y: (n, d) ndarrays
        n samples from d-dimensional multivariate normal distributions

    Returns:
    --------
    mi: float
        mutual information I(X, Y)
    """

    hx  = get_h_mvn(x)
    hy  = get_h_mvn(y)
    hxy = get_h_mvn(np.c_[x,y])
    mi = hx + hy - hxy

    # mi = 0.5 * (log(det(np.cov(x.T))) + log(det(np.cov(y.T))) - log(det(np.cov(np.c_[x,y].T))))

    return mi


@convert_vectors_to_2d_arrays_if_any
def get_pmi_mvn(x, y, z):
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
    ----------
    x, y, z: (n, d) ndarrays
        n samples from d-dimensional multivariate normal distributions

    Returns:
    --------
    pmi: float
        partial mutual information I(X;Y|Z)
    """

    d = x.shape[1]
    hz   = 0.5 * log((2 * np.pi * np.e)**d     * det(np.cov(z.T)))
    hxz  = 0.5 * log((2 * np.pi * np.e)**(2*d) * det(np.cov(x.T, y=z.T)))
    hyz  = 0.5 * log((2 * np.pi * np.e)**(2*d) * det(np.cov(y.T, y=z.T)))
    hxyz = 0.5 * log((2 * np.pi * np.e)**(3*d) * det(np.cov(np.c_[x,y,z].T)))

    pmi = hxz + hyz - hxyz - hz
    return pmi


@convert_vectors_to_2d_arrays_if_any
def get_h(x, k=1, norm='max', min_dist=0., workers=1):
    """
    Estimates the entropy H of a random variable x (in nats) based on
    the kth-nearest neighbour distances between point samples.

    @reference:
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9–16.

    Arguments:
    ----------
    x: (n, d) ndarray
        n samples from a d-dimensional multivariate distribution

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances

    min_dist: float (default 0.)
        minimum distance between data points;
        smaller distances will be capped using this value

    workers: int (default 1)
        number of workers to use for parallel processing in query;
        -1 uses all CPU threads

    Returns:
    --------
    h: float
        entropy H(X)
    """

    n, d = x.shape

    if norm == 'max': # max norm:
        p = np.inf
        log_c_d = 0 # volume of the d-dimensional unit ball
    elif norm == 'euclidean': # euclidean norm
        p = 2
        log_c_d = (d/2.) * log(np.pi) -log(gamma(d/2. +1))
    else:
        raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")

    kdtree = KDTree(x)

    # query all points -- k+1 as query point also in initial set
    # distances, _ = kdtree.query(x, k + 1, eps=0, p=norm)
    distances, _ = kdtree.query(x, k + 1, eps=0, p=p, workers=workers)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(log(2*distances)) # where did the 2 come from? radius -> diameter
    h = -digamma(k) + digamma(n) + log_c_d + (d / float(n)) * sum_log_dist

    return h


@convert_vectors_to_2d_arrays_if_any
def get_mi(x, y, k=1, normalize=None, norm='max', estimator='ksg', workers=1):
    """
    Estimates the mutual information (in nats) between two point clouds, x and y,
    in a D-dimensional space.

    I(X,Y) = H(X) + H(Y) - H(X,Y)

    @reference:
    Kraskov, Stoegbauer & Grassberger (2004). Estimating mutual information. PHYSICAL REVIEW E 69, 066138

    Arguments:
    ----------
    x, y: (n, d) ndarray
        n samples from d-dimensional multivariate distributions

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    normalize: function or None (default None)
        if a function, the data pre-processed with the function before the computation

    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances

    min_dist: float (default 0.)
        minimum distance between data points;
        smaller distances will be capped using this value

    estimator: 'ksg' or 'naive' (default 'ksg')
        'ksg'  : see Kraskov, Stoegbauer & Grassberger (2004) Estimating mutual information, eq(8).
        'naive': entropies are calculated individually using the Kozachenko-Leonenko estimator implemented in get_h()

    workers: int (default 1)
        number of workers to use for parallel processing in query;
        -1 uses all CPU threads

    Returns:
    --------
    mi: float
        mutual information I(X,Y)

    """

    if normalize:
        x = normalize(x)
        y = normalize(y)

    # construct state array for the joint process:
    xy = np.c_[x,y]

    if estimator == 'naive':
        # compute individual entropies
        hx  = get_h(x,  k=k, norm=norm)
        hy  = get_h(y,  k=k, norm=norm)
        hxy = get_h(xy, k=k, norm=norm)

        # compute mi
        mi = hx + hy - hxy

    elif estimator == 'ksg':

        # store data pts in kd-trees for efficient nearest neighbour computations
        # TODO: choose a better leaf size
        x_tree  = KDTree(x)
        y_tree  = KDTree(y)
        xy_tree = KDTree(xy)

        # kth nearest neighbour distances for every state
        if norm == 'max': # max norm:
            p = np.inf
        elif norm == 'euclidean': # euclidean norm
            p = 2
        else:
            raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")
        # query with k=k+1 to return the nearest neighbour, not counting the data point itself
        dist, _ = xy_tree.query(xy, k=k+1, p=p, workers=workers)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the x-subspace is strictly < epsilon
        # repeat for the y subspace
        n = len(x)
        nx = np.empty(n, dtype=int)
        ny = np.empty(n, dtype=int)
        for ii in range(n):
            nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1
            ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1

        mi = digamma(k) - np.mean(digamma(nx+1) + digamma(ny+1)) + digamma(n) # version (1)
        # mi = digamma(k) -1./k -np.mean(digamma(nx) + digamma(ny)) + digamma(n) # version (2)

    else:
        raise NotImplementedError("Estimator is one of 'naive', 'ksg'; currently: {}".format(estimator))

    return mi


@convert_vectors_to_2d_arrays_if_any
def get_pmi(x, y, z, k=1, normalize=None, norm='max', estimator='fp', workers=1):
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
    x, y, z: (n, d) ndarray
        n samples from d-dimensional multivariate distributions

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    normalize: function or None (default None)
        if a function, the data pre-processed with the function before the computation

    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances

    estimator: 'fp', 'ps' or 'naive' (default 'fp')
        'naive': entropies are calculated individually using the Kozachenko-Leonenko estimator implemented in get_h()
        'fp'   : Frenzel & Pombe estimator (effectively the KSG-estimator for mutual information)

    workers: int (default 1)
        number of workers to use for parallel processing in query;
        -1 uses all CPU threads

    Returns:
    --------
    pmi: float
        partial mutual information I(X,Y;Z)

    """

    if normalize:
        x = normalize(x)
        y = normalize(y)
        z = normalize(z)

    # construct state array for the joint processes:
    xz  = np.c_[x,z]
    yz  = np.c_[y,z]
    xyz = np.c_[x,y,z]

    if estimator == 'naive':
        # compute individual entropies
        # TODO: pass in min_dist
        hz   = get_h(z,   k=k, norm=norm)
        hxz  = get_h(xz,  k=k, norm=norm)
        hyz  = get_h(yz,  k=k, norm=norm)
        hxyz = get_h(xyz, k=k, norm=norm)

        pmi =  hxz + hyz - hxyz - hz

    elif estimator == 'fp':

        # construct k-d trees
        z_tree   = KDTree(z)
        xz_tree  = KDTree(xz)
        yz_tree  = KDTree(yz)
        xyz_tree = KDTree(xyz)

        # kth nearest neighbour distances for every state
        if norm == 'max': # max norm:
            p = np.inf
        elif norm == 'euclidean': # euclidean norm
            p = 2
        else:
            raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")
        # query with k=k+1 to return the nearest neighbour, not the data point itself
        dist, _ = xyz_tree.query(xyz, k=k+1, p=p, workers=workers)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the relevant subspace is strictly < epsilon
        n = len(x)
        nxz = np.empty(n, dtype=int)
        nyz = np.empty(n, dtype=int)
        nz  = np.empty(n, dtype=int)

        for ii in range(n):
            nz[ii]  = len( z_tree.query_ball_point( z_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1
            nxz[ii] = len(xz_tree.query_ball_point(xz_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1
            nyz[ii] = len(yz_tree.query_ball_point(yz_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1

        pmi = digamma(k) + np.mean(digamma(nz +1) -digamma(nxz +1) -digamma(nyz +1))

    elif estimator == 'ps':
        # I am fairly sure that this is the correct implementation of the estimator,
        # but the estimators is just crap.

        # construct k-d trees
        xz_tree  = KDTree(xz,  leafsize=2*k)
        yz_tree  = KDTree(yz,  leafsize=2*k)

        # determine k-nn distances
        n = len(x)
        rxz = np.empty(n, dtype=int)
        ryz = np.empty(n, dtype=int)

        if norm == 'max': # max norm:
            p = np.inf
        elif norm == 'euclidean': # euclidean norm
            p = 2
        else:
            raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")
        rxz, _ = xz_tree.query(xz, k=k+1, p=p, workers=workers) # +1 to account for distance to itself
        ryz, _ = yz_tree.query(xz, k=k+1, p=p, workers=workers) # +1 to account for distance to itself; xz NOT a typo

        pmi = yz.shape[1] * np.mean(log(ryz[:,-1]) - log(rxz[:,-1])) # + log(n) -log(n-1) -1.

    else:
        raise NotImplementedError("Estimator one of 'naive', 'fp', 'ps'; currently: {}".format(estimator))

    return pmi


@convert_vectors_to_2d_arrays_if_any
def get_imin(x1, x2, y, k=1, normalize=None, norm='max', workers=1):
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
    x1, x2, y: (n, d) ndarray
        n samples from d-dimensional multivariate distributions

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    normalize: function or None (default None)
        if a function, the data pre-processed with the function before the computation

    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances

    workers: int (default 1)
        number of workers to use for parallel processing in query;
        -1 uses all CPU threads

    Returns:
    --------
    i_min: float
        average specific information I_min(Y; X1, X2)

    """

    if normalize:
        y = normalize(y)

    if norm == 'max': # max norm:
        p = np.inf
    elif norm == 'euclidean': # euclidean norm
        p = 2
    else:
        raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")

    y_tree  = KDTree(y)

    n = len(y)
    i_spec = np.zeros((2, n))

    for jj, x in enumerate([x1, x2]):

        if normalize:
            x = normalize(x)

        # construct state array for the joint processes:
        xy = np.c_[x,y]

        # store data pts in kd-trees for efficient nearest neighbour computations
        # TODO: choose a better leaf size
        x_tree  = KDTree(x)
        xy_tree = KDTree(xy)

        # kth nearest neighbour distances for every state
        # query with k=k+1 to return the nearest neighbour, not counting the data point itself
        # dist, _ = xy_tree.query(xy, k=k+1, p=norm)
        dist, _ = xy_tree.query(xy, k=k+1, p=p, workers=workers)
        epsilon = dist[:, -1]

        # for each point, count the number of neighbours
        # whose distance in the x-subspace is strictly < epsilon
        # repeat for the y subspace
        nx = np.empty(n, dtype=int)
        ny = np.empty(n, dtype=int)
        for ii in range(n):
            nx[ii] = len(x_tree.query_ball_point(x_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1
            ny[ii] = len(y_tree.query_ball_point(y_tree.data[ii], r=epsilon[ii], p=p, workers=workers)) - 1

        i_spec[jj] = digamma(k) - digamma(nx+1) + digamma(ny+1) + digamma(n) # version (1)

    i_min = np.mean(np.min(i_spec, 0))

    return i_min


@convert_vectors_to_2d_arrays_if_any
def get_pid(x1, x2, y, k=1, normalize=None, norm='max'):

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
    x1, x2, y: (n, d) ndarray
        n samples from d-dimensional multivariate distributions

    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution

    normalize: function or None (default None)
        if a function, the data pre-processed with the function before the computation

    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances

    Returns:
    --------
    synergy: float
        information about Y encoded by the joint state of x1 and x2
    unique_x1: float
        information about Y encoded uniquely by x1
    unique_x2: float
        information about Y encoded uniquely by x2
    redundancy: float
        information about Y encoded by either x1 or x2

    """

    mi_x1y     = get_mi(x1,            y, k=k, normalize=normalize, norm=norm)
    mi_x2y     = get_mi(x2,            y, k=k, normalize=normalize, norm=norm)
    mi_x1x2y   = get_mi(np.c_[x1, x2], y, k=k, normalize=normalize, norm=norm)
    redundancy = get_imin(x1, x2,      y, k=k, normalize=normalize, norm=norm)

    unique_x1 = mi_x1y - redundancy
    unique_x2 = mi_x2y - redundancy
    synergy = mi_x1x2y - mi_x1y - mi_x2y + redundancy

    return synergy, unique_x1, unique_x2, redundancy
