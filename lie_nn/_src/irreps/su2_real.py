from dataclasses import dataclass
import itertools
from typing import Iterator

import numpy as np
from ..util import is_half_integer, is_integer, round_to_sqrt_rational

from ..irrep import TabulatedIrrep
from .su2 import SU2
from lie_nn import clebsch_gordan


def change_basis_real_to_complex(j: float) -> np.ndarray:
    if is_integer(j):
        l = int(j)
        # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
        q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
        for m in range(-l, 0):
            q[l + m, l + abs(m)] = 1 / np.sqrt(2)
            q[l + m, l - abs(m)] = -1j / np.sqrt(2)
        q[l, l] = 1
        for m in range(1, l + 1):
            q[l + m, l + abs(m)] = (-1) ** m / np.sqrt(2)
            q[l + m, l - abs(m)] = 1j * (-1) ** m / np.sqrt(2)
        return (-1j) ** l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    raise ValueError(f"j={j} is not an integer")


@dataclass(frozen=True)
class SU2Real(TabulatedIrrep):
    j: float  # j is a half-integer

    def __post_init__(rep):
        assert isinstance(rep.j, (float, int))
        assert is_half_integer(rep.j)
        assert rep.j >= 0

    def __mul__(rep1: "SU2Real", rep2: "SU2Real") -> Iterator["SU2Real"]:
        assert isinstance(rep2, SU2Real)
        return [
            SU2Real(j=float(j)) for j in np.arange(abs(rep1.j - rep2.j), rep1.j + rep2.j + 1, 1)
        ]

    @classmethod
    def clebsch_gordan(cls, rep1: "SU2Real", rep2: "SU2Real", rep3: "SU2Real") -> np.ndarray:
        from lie_nn import GenericRep

        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        if is_integer(rep1.j) and is_integer(rep2.j) and is_integer(rep3.j):
            C = SU2.clebsch_gordan(
                SU2(j=int(2 * rep1.j)), SU2(j=int(2 * rep2.j)), SU2(j=int(2 * rep3.j))
            )
            Q1 = change_basis_real_to_complex(rep1.j)
            Q2 = change_basis_real_to_complex(rep2.j)
            Q3 = change_basis_real_to_complex(rep3.j)
            C = np.einsum("ij,kl,mn,zikn->zjlm", Q1, Q2, np.conj(Q3.T), C)
        else:
            C = clebsch_gordan(
                GenericRep.from_rep(rep1), rep2, rep3, round_fn=round_to_sqrt_rational
            )

        assert np.all(np.abs(np.imag(C)) < 1e-5)
        return np.real(C)

    @property
    def dim(rep: "SU2Real") -> int:
        if is_integer(rep.j):
            return int(2 * rep.j + 1)
        else:
            return 2 * int(2 * rep.j + 1)

    def __lt__(rep1: "SU2Real", rep2: "SU2Real") -> bool:
        return rep1.j < rep2.j

    @classmethod
    def iterator(cls) -> Iterator["SU2Real"]:
        for tj in itertools.count(0):
            yield SU2Real(j=tj / 2)

    def continuous_generators(rep: "SU2Real") -> np.ndarray:
        X = SU2(j=int(2 * rep.j)).continuous_generators()

        if is_integer(rep.j):
            Q = change_basis_real_to_complex(rep.j)
            X = np.conj(Q.T) @ X @ Q

            assert np.all(np.abs(np.imag(X)) < 1e-5)
            return np.real(X)

        # convert complex array [d, i, j] to real array [d, i, 2, j, 2]
        return np.stack(
            [
                np.stack([np.real(X), -np.imag(X)], axis=3),
                np.stack([np.imag(X), np.real(X)], axis=3),
            ],
            axis=2,
        ).reshape((-1, rep.dim, rep.dim))

    def discrete_generators(rep: "SU2Real") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SU2.algebra()
