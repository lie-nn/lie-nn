import fractions
import itertools
from functools import partial
from typing import Iterator, List

import numpy as np

from ._abstract_rep import AbstractRep, static_jax_pytree
from ._su2 import SU2Rep


def is_integer(x: float) -> bool:
    return x == round(x)


def is_half_integer(x: float) -> bool:
    return 2 * x == round(2 * x)


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
            q[l + m, l + abs(m)] = (-1)**m / np.sqrt(2)
            q[l + m, l - abs(m)] = 1j * (-1)**m / np.sqrt(2)
        return (-1j)**l * q  # Added factor of 1j**l to make the Clebsch-Gordan coefficients real

    raise ValueError(f'j={j} is not an integer')


@partial(np.vectorize, otypes=[np.float64])
def round_to_sqrt_rational(x: float) -> float:
    sign = 1 if x >= 0 else -1
    return sign * fractions.Fraction(x**2).limit_denominator()**0.5


@static_jax_pytree
class SU2RealRep(AbstractRep):
    j: float  # j is a half-integer

    def __mul__(rep1: 'SU2RealRep', rep2: 'SU2RealRep') -> List['SU2RealRep']:
        assert isinstance(rep2, SU2RealRep)
        return [SU2RealRep(j=float(j)) for j in np.arange(abs(rep1.j - rep2.j), rep1.j + rep2.j + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: 'SU2RealRep', rep2: 'SU2RealRep', rep3: 'SU2RealRep') -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        if is_integer(rep1.j) and is_integer(rep2.j) and is_integer(rep3.j):
            C = SU2Rep.clebsch_gordan(SU2Rep(j=int(2 * rep1.j)), SU2Rep(j=int(2 * rep2.j)), SU2Rep(j=int(2 * rep3.j)))
            Q1 = change_basis_real_to_complex(rep1.j)
            Q2 = change_basis_real_to_complex(rep2.j)
            Q3 = change_basis_real_to_complex(rep3.j)
            C = np.einsum('ij,kl,mn,zikn->zjlm', Q1, Q2, np.conj(Q3.T), C)
        else:
            C = AbstractRep.clebsch_gordan(rep1, rep2, rep3)
            C = round_to_sqrt_rational(C)

        assert np.all(np.abs(np.imag(C)) < 1e-5)
        return np.real(C)

    @property
    def dim(rep: 'SU2RealRep') -> int:
        if is_integer(rep.j):
            return int(2 * rep.j + 1)
        else:
            return 2 * int(2 * rep.j + 1)

    @classmethod
    def iterator(cls) -> Iterator['SU2RealRep']:
        for tj in itertools.count(0):
            yield SU2RealRep(j=tj / 2)

    def continuous_generators(rep: 'SU2RealRep') -> np.ndarray:
        X = SU2Rep(j=int(2 * rep.j)).continuous_generators()

        if is_integer(rep.j):
            Q = change_basis_real_to_complex(rep.j)
            X = np.conj(Q.T) @ X @ Q

            assert np.all(np.abs(np.imag(X)) < 1e-5)
            return np.real(X)

        # convert complex array [d, i, j] to real array [d, i, 2, j, 2]
        return np.stack([
            np.stack([np.real(X), -np.imag(X)], axis=3),
            np.stack([np.imag(X), np.real(X)], axis=3),
        ], axis=2).reshape((-1, rep.dim, rep.dim))

    def discrete_generators(rep: 'SU2RealRep') -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    @classmethod
    def algebra(cls) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SU2Rep.algebra()
