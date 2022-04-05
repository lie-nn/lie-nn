import itertools
from typing import Iterator, List

import chex
import jax.numpy as jnp

from . import AbstractRep


@chex.dataclass(frozen=True)
class Rep(AbstractRep):
    j: float  # half integer

    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        return [Rep(j=j) for j in range(abs(ir1.j - ir2.j), ir1.j + ir2.j + 1)]

    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        pass

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
