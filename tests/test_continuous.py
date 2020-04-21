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

from continuous import *


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
    for ii, jj in itertools.product(list(range(total_rvs)), repeat=2):
        if ii != jj:
            sigma[d*ii:d*(ii+1), d*jj:d*(jj+1)] *= scale_sigma_offdiagonal_by

    # get samples
    samples = multivariate_normal(mu, sigma).rvs(total_samples)

    return [samples[:,ii*d:(ii+1)*d] for ii in range(total_rvs)]


def test_get_h(k=5, norm='max'):
    X, = get_mvn_data(total_rvs=1,
                      dimensionality=2,
                      scale_sigma_offdiagonal_by=1.,
                      total_samples=1000)

    analytic = get_h_mvn(X)
    kozachenko = get_h(X, k=k, norm=norm)

    print("analytic result: {:.5f}".format(analytic))
    print("K-L estimator:   {:.5f}".format(kozachenko))
    assert np.isclose(analytic, kozachenko, rtol=0.1, atol=0.1), "K-L estimate strongly differs from analytic expectation!"


def test_get_h_1d(k=5, norm='max'):
    X = np.random.randn(1000)

    analytic = get_h_mvn(X)
    kozachenko = get_h(x, k=k, norm=norm)

    print("analytic result: {:.5f}".format(analytic))
    print("K-L estimator:   {:.5f}".format(kozachenko))
    assert np.isclose(analytic, kozachenko, rtol=0.1, atol=0.1), "K-L estimate strongly differs from analytic expectation!"


def test_get_mi(k=5, normalize=None, norm='max'):

    X, Y = get_mvn_data(total_rvs=2,
                        dimensionality=2,
                        scale_sigma_offdiagonal_by=1., # 0.1, 0.
                        total_samples=10000)

    # solutions
    analytic = get_mi_mvn(X, Y)
    naive = get_mi(X, Y, k=k, normalize=normalize, norm=norm, estimator='naive')
    ksg   = get_mi(X, Y, k=k, normalize=normalize, norm=norm, estimator='ksg')

    print("analytic result:  {:.5f}".format(analytic))
    print("naive estimator:  {:.5f}".format(naive))
    print("KSG estimator:    {:.5f}".format(ksg))

    print("naive - analytic: {:.5f}".format(naive - analytic))
    print("ksg   - analytic: {:.5f}".format(ksg   - analytic))

    print("naive / analytic: {:.5f}".format(naive / analytic))
    print("ksg   / analytic: {:.5f}".format(ksg   / analytic))

    # for automated testing:
    assert np.isclose(analytic, naive, rtol=0.1, atol=0.1), "Naive MI estimate strongly differs from expectation!"
    assert np.isclose(analytic, ksg,   rtol=0.1, atol=0.1), "KSG MI estimate strongly differs from expectation!"


def test_get_pmi(k=5, normalize=None, norm='max'):

    X, Y, Z = get_mvn_data(total_rvs=3,
                           dimensionality=2,
                           scale_sigma_offdiagonal_by=1.,
                           total_samples=10000)

    # solutions
    analytic = get_pmi_mvn(X, Y, Z)
    naive    = get_pmi(X, Y, Z, k=k, normalize=normalize, norm=norm, estimator='naive')
    fp       = get_pmi(X, Y, Z, k=k, normalize=normalize, norm=norm, estimator='fp')

    print("analytic result : {:.5f}".format(analytic))
    print("naive estimator : {:.5f}".format(naive))
    print("FP estimator    : {:.5f}".format(fp))

    # for automated testing:
    assert np.isclose(analytic, naive, rtol=0.5, atol=0.5), "Naive MI estimate strongly differs from expectation!"
    assert np.isclose(analytic, fp,    rtol=0.5, atol=0.5), "FP MI estimate strongly differs from expectation!"


def test_get_pid(k=5, normalize=None, norm='max'):
    # rdn -> only redundant information

    # unq -> only unique information

    # xor -> only synergistic information

    pass
