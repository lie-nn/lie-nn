from dataclasses import dataclass
import itertools
from typing import Iterator

import numpy as np

from ..irrep import TabulatedIrrep
from .su2 import SU2


def change_basis_real_to_complex(l: int) -> np.ndarray:
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


@dataclass(frozen=True)
class SO3(TabulatedIrrep):
    l: int

    @classmethod
    def from_string(cls, s: str) -> "SO3":
        return cls(l=int(s))

    def __mul__(rep1: "SO3", rep2: "SO3") -> Iterator["SO3"]:
        assert isinstance(rep2, SO3)
        return [SO3(l=l) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: "SO3", rep2: "SO3", rep3: "SO3") -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        C = SU2.clebsch_gordan(SU2(j=2 * rep1.l), SU2(j=2 * rep2.l), SU2(j=2 * rep3.l))
        Q1 = change_basis_real_to_complex(rep1.l)
        Q2 = change_basis_real_to_complex(rep2.l)
        Q3 = change_basis_real_to_complex(rep3.l)
        C = np.einsum("ij,kl,mn,zikn->zjlm", Q1, Q2, np.conj(Q3.T), C)

        # make it real
        assert np.all(np.abs(np.imag(C)) < 1e-5)
        C = np.real(C)

        return C

    @property
    def dim(rep: "SO3") -> int:
        return 2 * rep.l + 1

    def __lt__(rep1: "SO3", rep2: "SO3") -> bool:
        return rep1.l < rep2.l

    @classmethod
    def iterator(cls) -> Iterator["SO3"]:
        for l in itertools.count(0):
            yield SO3(l=l)

    def continuous_generators(rep: "SO3") -> np.ndarray:
        X = SU2(j=2 * rep.l).continuous_generators()
        Q = change_basis_real_to_complex(rep.l)
        X = np.conj(Q.T) @ X @ Q
        assert np.all(np.abs(np.imag(X)) < 1e-5)
        return np.real(X)

    def discrete_generators(rep: "SO3") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SU2.algebra(rep)
