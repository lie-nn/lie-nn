import itertools
from math import factorial
from typing import Iterator, List
from flax import struct

import jax.numpy as jnp
import numpy as np

from ._abstract_rep import AbstractRep


@struct.dataclass
class SU2Rep(AbstractRep):
    j: int = struct.field(pytree_node=False)

    def __mul__(rep1: 'SU2Rep', rep2: 'SU2Rep') -> List['SU2Rep']:
        return [SU2Rep(j=j) for j in range(abs(rep1.j - rep2.j), rep1.j + rep2.j + 1, 2)]

    @classmethod
    def clebsch_gordan(cls, rep1: 'SU2Rep', rep2: 'SU2Rep', rep3: 'SU2Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        if rep3 in rep1 * rep2:
            return clebsch_gordanSU2mat(rep1.j / 2, rep2.j / 2, rep3.j / 2)[None]
        else:
            return jnp.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: 'SU2Rep') -> int:
        return rep.j + 1

    @classmethod
    def iterator(cls) -> Iterator['SU2Rep']:
        for j in itertools.count(0):
            yield SU2Rep(j=j)

    def discrete_generators(rep: 'SU2Rep') -> jnp.ndarray:
        return jnp.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: 'SU2Rep') -> jnp.ndarray:
        hj = rep.j / 2.0  # half-j
        m = jnp.arange(-hj, hj)
        raising = jnp.diag(-jnp.sqrt(hj * (hj + 1) - m * (m + 1)), k=-1)

        m = jnp.arange(-hj + 1, hj + 1)
        lowering = jnp.diag(jnp.sqrt(hj * (hj + 1) - m * (m - 1)), k=1)

        m = jnp.arange(-hj, hj + 1)
        return jnp.stack([
            0.5j * (raising - lowering),  # y (usually)
            jnp.diag(1j * m),  # z (usually)
            0.5 * (raising + lowering),  # x (usually)
        ], axis=0)

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return jnp.array([
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            [
                [0, 0, -1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 0.0],
            ],
        ])


# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

def clebsch_gordanSU2mat(j1, j2, j3):
    """Calculates the Clebsch-Gordon matrix
    for SU(2) coupling j1 and j2 to give j3.
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    Returns
    -------
    cg_matrix : numpy.array
        Requested Clebsch-Gordan matrix.
    """
    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1), int(2 * j3 + 1)))
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)
                        ] = clebsch_gordanSU2coeffs((j1, m1), (j2, m2), (j3, m1 + m2))
    return np.array(mat)


def clebsch_gordanSU2coeffs(idx1, idx2, idx3):
    """Calculates the Clebsch-Gordon coefficient
    for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3).
    Parameters
    ----------
    j1 : float
        Total angular momentum 1.
    j2 : float
        Total angular momentum 2.
    j3 : float
        Total angular momentum 3.
    m1 : float
        z-component of angular momentum 1.
    m2 : float
        z-component of angular momentum 2.
    m3 : float
        z-component of angular momentum 3.
    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.
    """
    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        assert n == round(n)
        return factorial(round(n))

    C = np.sqrt((2.0 * j3 + 1.0) * f(j3 + j1 - j2) * f(j3 - j1 + j2) * f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3) /  # noqa: W504
                (f(j1 + j2 + j3 + 1) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / f(v) * f(j2 + j3 + m1 - v) * f(j1 - m1 + v) / \
            f(j3 - j1 + j2 - v) / \
            f(j3 + m3 - v) / \
            f(v + j1 - j2 - m3)
    C = C * S
    return C
