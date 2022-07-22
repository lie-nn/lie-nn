from dataclasses import dataclass
import itertools
from typing import Iterator

import numpy as np

from .. import Irrep
from ._su2 import SU2Rep


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
class SO3Rep(Irrep):
    l: int

    def __mul__(rep1: "SO3Rep", rep2: "SO3Rep") -> Iterator["SO3Rep"]:
        assert isinstance(rep2, SO3Rep)
        return [SO3Rep(l=l) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: "SO3Rep", rep2: "SO3Rep", rep3: "SO3Rep") -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        C = SU2Rep.clebsch_gordan(SU2Rep(j=2 * rep1.l), SU2Rep(j=2 * rep2.l), SU2Rep(j=2 * rep3.l))
        Q1 = change_basis_real_to_complex(rep1.l)
        Q2 = change_basis_real_to_complex(rep2.l)
        Q3 = change_basis_real_to_complex(rep3.l)
        C = np.einsum("ij,kl,mn,zikn->zjlm", Q1, Q2, np.conj(Q3.T), C)

        # make it real
        assert np.all(np.abs(np.imag(C)) < 1e-5)
        C = np.real(C)

        return C

    @property
    def dim(rep: "SO3Rep") -> int:
        return 2 * rep.l + 1

    @classmethod
    def iterator(cls) -> Iterator["SO3Rep"]:
        for l in itertools.count(0):
            yield SO3Rep(l=l)

    def continuous_generators(rep: "SO3Rep") -> np.ndarray:
        X = SU2Rep(j=2 * rep.l).continuous_generators()
        Q = change_basis_real_to_complex(rep.l)
        X = np.conj(Q.T) @ X @ Q
        assert np.all(np.abs(np.imag(X)) < 1e-5)
        return np.real(X)

    def discrete_generators(rep: "SO3Rep") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def algebra(rep) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SU2Rep.algebra(rep)
