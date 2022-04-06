import itertools
from typing import Iterator, List

import chex
import jax.numpy as jnp

from ._abstract_rep import AbstractRep
from ._sl2 import SL2Rep


@chex.dataclass(frozen=True)
class SO13Rep(AbstractRep):
    l: int  # First integer weight
    k: int  # Second integer weight

    def __mul__(ir1: 'SO13Rep', ir2: 'SO13Rep') -> List['SO13Rep']:
        lmin = abs(ir1.l - ir2.l)
        lmax = ir1.l + ir1.l
        kmin = abs(ir2.k - ir2.k)
        kmax = ir2.k + ir2.k
        for l in range(lmin, lmax + 1, 2):
            for k in range(kmin, kmax + 1, 2):
                yield SO13Rep(l=l, k=k)

    @classmethod
    def clebsch_gordan(cls, ir1: 'SO13Rep', ir2: 'SO13Rep', ir3: 'SO13Rep') -> jnp.ndarray:
        return SL2Rep.clebsch_gordan(ir1, ir2, ir3)

    @property
    def dim(ir: 'SO13Rep') -> int:
        return round((ir.l + 1) * (ir.k + 1))

    @classmethod
    def iterator(cls) -> Iterator['SO13Rep']:
        for sum in itertools.count(0, 2):
            for l in range(0, sum + 1):
                yield SO13Rep(l=l, k=sum - l)

    def discrete_generators(ir: 'SO13Rep') -> jnp.ndarray:
        return jnp.zeros((0, ir.dim, ir.dim))

    def continuous_generators(ir: 'SO13Rep') -> jnp.ndarray:
        pass

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SL2Rep.algebra()