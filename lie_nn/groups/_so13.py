import itertools
from typing import Iterator, List

import jax.numpy as jnp

from ._abstract_rep import AbstractRep, static_jax_pytree
from ._sl2 import SL2Rep


@static_jax_pytree
class SO13Rep(AbstractRep):
    l: int  # First integer weight
    k: int  # Second integer weight

    def __mul__(rep1: "SO13Rep", rep2: "SO13Rep") -> List["SO13Rep"]:
        assert isinstance(rep2, SO13Rep)
        lmin = abs(rep1.l - rep2.l)
        lmax = rep1.l + rep1.l
        kmin = abs(rep2.k - rep2.k)
        kmax = rep2.k + rep2.k
        for l in range(lmin, lmax + 1, 2):
            for k in range(kmin, kmax + 1, 2):
                yield SO13Rep(l=l, k=k)

    @classmethod
    def clebsch_gordan(cls, rep1: "SO13Rep", rep2: "SO13Rep", rep3: "SO13Rep") -> jnp.ndarray:
        return SL2Rep.clebsch_gordan(rep1, rep2, rep3)

    @property
    def dim(rep: "SO13Rep") -> int:
        return round((rep.l + 1) * (rep.k + 1))

    @classmethod
    def iterator(cls) -> Iterator["SO13Rep"]:
        for sum in itertools.count(0, 2):
            for l in range(0, sum + 1):
                yield SO13Rep(l=l, k=sum - l)

    def discrete_generators(rep: "SO13Rep") -> jnp.ndarray:
        return jnp.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SO13Rep") -> jnp.ndarray:
        return SL2Rep.continuous_generators(rep)

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SL2Rep.algebra()
