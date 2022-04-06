import itertools
from typing import Iterator, List

import chex
import jax.numpy as jnp

from . import AbstractRep
from .o3_real import Rep as O3Rep


@chex.dataclass(frozen=True)
class Rep(AbstractRep):
    l: int  # non-negative integer
    p: int  # 1 or -1

    def __mul__(rep1: 'Rep', rep2: 'Rep') -> List['Rep']:
        p = rep1.p * rep2.p
        return [Rep(l=l, p=p) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: 'Rep', rep2: 'Rep', rep3: 'Rep') -> jnp.ndarray:
        if rep1.p * rep2.p == rep3.p:
            return O3Rep.clebsch_gordan(rep1, rep2, rep3)
        else:
            return jnp.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: 'Rep') -> int:
        return 2 * rep.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        for l in itertools.count(0):
            yield Rep(l=l, p=1)
            yield Rep(l=l, p=-1)

    def continuous_generators(rep: 'Rep') -> jnp.ndarray:
        return O3Rep(l=rep.l).continuous_generators()

    def discrete_generators(rep: 'Rep') -> jnp.ndarray:
        return rep.p * jnp.eye(rep.dim)[None]

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        return O3Rep.algebra()
