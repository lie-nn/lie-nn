import itertools
from math import factorial
from typing import Iterator, List

import numpy as np

import chex
import jax.numpy as jnp

from . import AbstractRep


@chex.dataclass(frozen=True)
class Rep(AbstractRep):
    j: float  # half integer

    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        return [Rep(j=float(j)) for j in np.arange(abs(ir1.j - ir2.j), ir1.j + ir2.j + 1, 1.0)]

    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        if ir3 in ir1 * ir2:
            return clebsch_gordanSU2mat(ir1.j, ir2.j, ir3.j)[None]
        else:
            return jnp.zeros((0, ir1.dim, ir2.dim, ir3.dim))

    @property
    def dim(ir: 'Rep') -> int:
        return round(2 * ir.j + 1)

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        for twice_j in itertools.count(0):
            yield Rep(j=twice_j / 2)

    def discrete_generators(ir: 'Rep') -> jnp.ndarray:
        return jnp.zeros((0, ir.dim, ir.dim))

    def continuous_generators(ir: 'Rep') -> jnp.ndarray:
        j = ir.j
        m = jnp.arange(-j, j)
        raising = jnp.diag(-jnp.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

        m = jnp.arange(-j + 1, j + 1)
        lowering = jnp.diag(jnp.sqrt(j * (j + 1) - m * (m - 1)), k=1)

        m = jnp.arange(-j, j + 1)
        return jnp.stack([
            0.5 * (raising + lowering),
            0.5j * (raising - lowering),
            jnp.diag(1j * m)
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

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) * factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) * factorial(j3 + m3) * factorial(j3 - m3) /  # noqa: W504
                (factorial(j1 + j2 + j3 + 1) * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / \
            factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C
