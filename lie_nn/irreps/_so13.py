import itertools
from typing import Iterator

import numpy as np

from .. import Irrep, static_jax_pytree
from ._sl2 import SL2Rep


@static_jax_pytree
class SO13Rep(Irrep):  # TODO: think if this class shoulb be a subclass of SL2Rep
    l: int  # First integer weight
    k: int  # Second integer weight

    def __post_init__(rep):
        assert isinstance(rep.l, int)
        assert isinstance(rep.k, int)
        assert rep.l >= 0
        assert rep.k >= 0
        assert (rep.l + rep.k) % 2 == 0

    def __mul__(rep1: "SO13Rep", rep2: "SO13Rep") -> Iterator["SO13Rep"]:
        for rep in SL2Rep.__mul__(rep1, rep2):
            yield SO13Rep(l=rep.l, k=rep.k)

    @classmethod
    def clebsch_gordan(cls, rep1: "SO13Rep", rep2: "SO13Rep", rep3: "SO13Rep") -> np.ndarray:
        return SL2Rep.clebsch_gordan(rep1, rep2, rep3)

    @property
    def dim(rep: "SO13Rep") -> int:
        return SL2Rep(l=rep.l, k=rep.k).dim

    @classmethod
    def iterator(cls) -> Iterator["SO13Rep"]:
        for sum in itertools.count(0, 2):
            for l in range(0, sum + 1):
                yield SO13Rep(l=l, k=sum - l)

    def discrete_generators(rep: "SO13Rep") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SO13Rep") -> np.ndarray:
        return SL2Rep.continuous_generators(rep)

    @classmethod
    def algebra(cls) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return SL2Rep.algebra()
