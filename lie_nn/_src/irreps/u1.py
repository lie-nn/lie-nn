import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..rep import TabulatedIrrep


@dataclass(frozen=True)
class U1(TabulatedIrrep):
    m: int

    def __post_init__(self):
        assert isinstance(self.m, int)
        assert self.m >= 0

    @classmethod
    def from_string(cls, string: str) -> "U1":
        return cls(m=int(string))

    def __mul__(rep1: "U1", rep2: "U1") -> Iterator["U1"]:
        assert isinstance(rep2, U1)
        return [U1(m=rep1.m + rep2.m)]

    @classmethod
    def clebsch_gordan(cls, rep1: "U1", rep2: "U1", rep3: "U1") -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        if rep3 in rep1 * rep2:
            return np.ones((1, 1, 1, 1))
        else:
            return np.zeros((0, 1, 1, 1))

    @property
    def dim(rep: "U1") -> int:
        return 1

    def is_scalar(rep: "U1") -> bool:
        """Equivalent to ``j == 0``"""
        return rep.m == 0

    def __lt__(rep1: "U1", rep2: "U1") -> bool:
        return rep1.m < rep2.m

    @classmethod
    def iterator(cls) -> Iterator["U1"]:
        for m in itertools.count(0):
            yield U1(m=m)

    def discrete_generators(rep: "U1") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "U1") -> np.ndarray:
        return np.array([[[1j * rep.m]]])

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return np.zeros((1, 1, 1))
