import itertools
from typing import Iterator, List

import chex
import jax.numpy as jnp

from . import AbstractRep


@chex.dataclass(frozen=True)
class Rep(AbstractRep):
    l: float  # half integer
    k: float

    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        lmin = abs(ir1.l - ir2.l)
        lmax = ir1.l + ir1.l
        kmin = abs(ir2.k - ir2.k)
        kmax = abs(ir2.k + ir2.k)
        for l in range(lmin, lmax + 1, 2):
            for k in range(kmin, kmax + 1, 2):
                yield Rep(l=l, k=k)

    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        pass

    @property
    def dim(ir: 'Rep') -> int:
        return round((ir.l + 1) * (ir.k + 1))

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        for l, k in zip(itertools.count(0)):
            if (l + k) % 2 == 0:
                yield Rep(l=l, k=k)

    def discrete_generators(ir: 'Rep') -> jnp.ndarray:
        return jnp.zeros((0, ir.dim, ir.dim))

    def continuous_generators(ir: 'Rep') -> jnp.ndarray:
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
