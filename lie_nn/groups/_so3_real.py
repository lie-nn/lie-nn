import itertools
from typing import Iterator, List

from flax import struct
import numpy as np

from ._abstract_rep import AbstractRep
from ._su2 import SU2Rep


def change_basis_real_to_complex(l: int) -> np.ndarray:
    q = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(1, l + 1):
        w = 1j**(-m) / np.sqrt(2)

        q[l - m, l - m] = -1j * w
        q[l - m, l + m] = w
        q[l + m, l - m] = 1j * w
        q[l + m, l + m] = w

    q[l, l] = 1
    return q


@struct.dataclass
class SO3Rep(AbstractRep):
    l: int = struct.field(pytree_node=False)

    def __mul__(rep1: 'SO3Rep', rep2: 'SO3Rep') -> List['SO3Rep']:
        return [SO3Rep(l=l) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: 'SO3Rep', rep2: 'SO3Rep', rep3: 'SO3Rep') -> np.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        C = SU2Rep.clebsch_gordan(SU2Rep(j=2 * rep1.l), SU2Rep(j=2 * rep2.l), SU2Rep(j=2 * rep3.l))
        Q1 = change_basis_real_to_complex(rep1.l)
        Q2 = change_basis_real_to_complex(rep2.l)
        Q3 = change_basis_real_to_complex(rep3.l)
        C = np.einsum('ij,kl,mn,zikn->zjlm', Q1, Q2, np.conj(Q3.T), C)

        # make it real
        C = 1j**(rep1.l + rep2.l + rep3.l) * C
        # assert np.all(np.abs(np.imag(C)) < 1e-5)
        C = np.real(C)

        # normalization
        C = C / np.linalg.norm(C)
        return C

    @property
    def dim(rep: 'SO3Rep') -> int:
        return 2 * rep.l + 1

    @classmethod
    def iterator(cls) -> Iterator['SO3Rep']:
        for l in itertools.count(0):
            yield SO3Rep(l=l)

    def continuous_generators(rep: 'SO3Rep') -> np.ndarray:
        X = SU2Rep(j=2 * rep.l).continuous_generators()
        Q = change_basis_real_to_complex(rep.l)
        X = np.conj(Q.T) @ X @ Q
        # assert np.max(np.abs(np.imag(X))) < 1e-5
        return np.real(X)

    def discrete_generators(rep: 'SO3Rep') -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    @classmethod
    def algebra(cls) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SU2Rep.algebra()
