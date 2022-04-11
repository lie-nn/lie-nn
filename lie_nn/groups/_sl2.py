import itertools
from typing import Iterator, List

import jax.numpy as jnp
import numpy as np
from lie_nn.groups._su2 import clebsch_gordanSU2mat

from ._abstract_rep import AbstractRep, static_jax_pytree


@static_jax_pytree
class SL2Rep(AbstractRep):
    l: int  # First integer weight
    k: int  # Second integer weight

    def __mul__(rep1: 'SL2Rep', rep2: 'SL2Rep') -> List['SL2Rep']:
        assert isinstance(rep2, SL2Rep)
        lmin = abs(rep1.l - rep2.l)
        lmax = rep1.l + rep1.l
        kmin = abs(rep2.k - rep2.k)
        kmax = rep2.k + rep2.k
        for l in range(lmin, lmax + 1):
            for k in range(kmin, kmax + 1):
                yield SL2Rep(l=l, k=k)

    @classmethod
    def clebsch_gordan(cls, rep1: 'SL2Rep', rep2: 'SL2Rep', rep3: 'SL2Rep') -> jnp.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        if rep3 in rep1 * rep2:
            return clebsch_gordansl2mat((rep1.l, rep1.k), (rep2.l, rep2.k), (rep3.l, rep3.k))
        else:
            return jnp.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: 'SL2Rep') -> int:
        return (rep.l + 1) * (rep.k + 1)

    @classmethod
    def iterator(cls) -> Iterator['SL2Rep']:
        for sum in itertools.count(0):
            for l in range(0, sum + 1):
                yield SL2Rep(l=l, k=sum - l)

    def discrete_generators(rep: 'SL2Rep') -> jnp.ndarray:
        return jnp.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: 'SL2Rep') -> jnp.ndarray:
        pass

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return jnp.array([[[0., 1., 0., 0.],
                           [1., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.]],

                          [[0., 0., 1., 0.],
                           [0., 0., 0., 0.],
                           [1., 0., 0., 0.],
                           [0., 0., 0., 0.]],

                          [[0., 0., 0., 1.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [1., 0., 0., 0.]],

                          [[0., 0., 0., 0.],
                           [0., 0., -1., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 0., 0.]],

                          [[0., 0., 0., 0.],
                           [0., 0., 0., -1.],
                           [0., 0., 0., 0.],
                           [0., 1., 0., 0.]],

                          [[0., 0., 0., 0.],
                           [0., 0., 0., 0.],
                           [0., 0., 0., -1.],
                           [0., 0., 1., 0.]]])

# From Lorentz group equivariant network Bogatskiy

###############################################################################
#  Copyright (C) 2019, The University of Chicago, Brandon Anderson,
#  Alexander Bogatskiy, David Miller, Jan Offermann, and Risi Kondor.
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
###############################################################################


def clebsch_gordansl2mat(rep1, rep2, rep3, fastcgmat=clebsch_gordanSU2mat):
    """
    Calculates the Clebsch-Gordon matrix
    for SL(2,C) coupling (l1,l1) and (l2,l2) to give (l3,k3).
    Parameters
    ----------
    rep1 : Tuple(int)
        Weights (l1,k1) of the first representation.
    rep2 : Tuple(int)
        Weights (l2,k2) of the first representation.
    rep3 : Tuple(int)
        Weights (l,k) of the first representation.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    l1, k1 = rep1
    l2, k2 = rep2
    l, k = rep3
    B1 = np.concatenate([fastcgmat(l / 2, k / 2, i / 2)
                         for i in range(abs(l - k), l + k + 1, 2)], axis=-1)
    B2a = fastcgmat(l1 / 2, l2 / 2, l / 2)
    B2b = fastcgmat(k1 / 2, k2 / 2, k / 2)
    B3a = np.concatenate([fastcgmat(l1 / 2, k1 / 2, i1 / 2)
                          for i1 in range(abs(l1 - k1), l1 + k1 + 1, 2)], axis=-1)
    B3b = np.concatenate([fastcgmat(l2 / 2, k2 / 2, i2 / 2)
                          for i2 in range(abs(l2 - k2), l2 + k2 + 1, 2)], axis=-1)
    cg_matrix = np.einsum('cab', np.einsum('abc,dea,ghb,dgk,ehn', B1, B2a, B2b, B3a, B3b))
    return cg_matrix
