import itertools
import re
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..irrep import TabulatedIrrep
from ..util import permutation_sign, vmap
from .su2 import SU2, clebsch_gordanSU2mat


@dataclass(frozen=True)
class SL2C(TabulatedIrrep):
    l: int  # First integer weight
    k: int  # Second integer weight

    def __post_init__(rep):
        assert isinstance(rep.l, int)
        assert isinstance(rep.k, int)
        assert rep.l >= 0
        assert rep.k >= 0

    @classmethod
    def from_string(cls, s: str) -> "SL2C":
        m = re.match(r"\((\d+),(\d+)\)", s.strip())
        assert m is not None
        return cls(l=int(m.group(1)), k=int(m.group(2)))

    def __mul__(rep1: "SL2C", rep2: "SL2C") -> Iterator["SL2C"]:
        for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 2):
            for k in range(abs(rep1.k - rep2.k), rep1.k + rep2.k + 1, 2):
                yield SL2C(l=l, k=k)

    @classmethod
    def clebsch_gordan(cls, rep1: "SL2C", rep2: "SL2C", rep3: "SL2C") -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        if rep3 in rep1 * rep2:
            return clebsch_gordansl2mat((rep1.l, rep1.k), (rep2.l, rep2.k), (rep3.l, rep3.k))[None]
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: "SL2C") -> int:
        return (rep.l + 1) * (rep.k + 1)

    def is_scalar(rep: "SL2C") -> bool:
        """Equivalent to ``l == 0 and k == 0``"""
        return rep.l == 0 and rep.k == 0

    def __lt__(rep1: "SL2C", rep2: "SL2C") -> bool:
        return (rep1.l + rep1.k, rep1.l) < (rep2.l + rep2.k, rep2.l)

    @classmethod
    def iterator(cls) -> Iterator["SL2C"]:
        for sum in itertools.count(0):
            for l in range(0, sum + 1):
                yield SL2C(l=l, k=sum - l)

    def discrete_generators(rep: "SL2C") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SL2C") -> np.ndarray:
        def id_like(x):
            return np.eye(x.shape[0])

        def kron_add(x, y):
            return np.kron(x, id_like(y)) + np.conj(np.kron(id_like(x), y))

        Xl = SU2(j=rep.l).continuous_generators()
        Xk = SU2(j=rep.k).continuous_generators()
        real = vmap(kron_add)(Xl, Xk)
        imag = vmap(kron_add)(Xl, -Xk)
        X = np.concatenate([real, imag], axis=0)
        C = SL2C.clebsch_gordan(
            SL2C(l=rep.l, k=0), SL2C(l=0, k=rep.k), SL2C(l=rep.l, k=rep.k)
        ).reshape(
            rep.dim, rep.dim
        )  # [d, d]
        return C.T @ X @ np.conj(C)

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        algebra = np.zeros((6, 6, 6))

        # for generators X_0, X_1, X_2, Y_0, Y_1, Y_2
        for i, j, k in itertools.permutations((0, 1, 2)):
            algebra[i, j, k] = permutation_sign((i, j, k))  # [X_i, X_j] = eps_ijk X_k
            algebra[3 + i, 3 + j, k] = permutation_sign((i, j, k))  # [Y_i, Y_j] = eps_ijk X_k
            algebra[i, 3 + j, 3 + k] = permutation_sign((i, j, k))  # [X_i, Y_j] = eps_ijk Y_k
            algebra[3 + i, j, 3 + k] = permutation_sign((i, j, k))  # [Y_i, X_j] = eps_ijk Y_k

        return algebra


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
    B1 = np.concatenate(
        [fastcgmat(l / 2, k / 2, i / 2) for i in range(abs(l - k), l + k + 1, 2)], axis=-1
    )
    B2a = fastcgmat(l1 / 2, l2 / 2, l / 2)
    B2b = fastcgmat(k1 / 2, k2 / 2, k / 2)
    B3a = np.concatenate(
        [fastcgmat(l1 / 2, k1 / 2, i1 / 2) for i1 in range(abs(l1 - k1), l1 + k1 + 1, 2)], axis=-1
    )
    B3b = np.concatenate(
        [fastcgmat(l2 / 2, k2 / 2, i2 / 2) for i2 in range(abs(l2 - k2), l2 + k2 + 1, 2)], axis=-1
    )
    cg_matrix = np.einsum("cab", np.einsum("abc,dea,ghb,dgk,ehn", B1, B2a, B2b, B3a, B3b))
    return cg_matrix
