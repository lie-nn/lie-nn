import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..irrep import TabulatedIrrep
from .so3_real import SO3


@dataclass(frozen=True)
class O3(TabulatedIrrep):
    l: int  # non-negative integer
    p: int  # 1 or -1

    @classmethod
    def from_string(cls, s: str) -> "O3":
        s = s.strip()
        l = int(s[:-1])
        p = {"e": 1, "o": -1}[s[-1]]
        return cls(l=l, p=p)

    def __mul__(rep1: "O3", rep2: "O3") -> Iterator["O3"]:
        assert isinstance(rep2, O3)
        p = rep1.p * rep2.p
        return [O3(l=l, p=p) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: "O3", rep2: "O3", rep3: "O3") -> np.ndarray:
        if rep1.p * rep2.p == rep3.p:
            return SO3.clebsch_gordan(rep1, rep2, rep3)
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: "O3") -> int:
        return 2 * rep.l + 1

    def is_scalar(rep: "O3") -> bool:
        """Equivalent to ``l == 0``"""
        return rep.l == 0

    def __lt__(rep1: "O3", rep2: "O3") -> bool:
        # scalar, speudo-scalar, vector, pseudo-vector, tensor, pseudo-tensor, ...
        return (rep1.l, -((-1) ** rep1.l) * rep1.p) < (rep2.l, -((-1) ** rep2.l) * rep2.p)

    @classmethod
    def iterator(cls) -> Iterator["O3"]:
        for l in itertools.count(0):
            yield O3(l=l, p=1 * (-1) ** l)
            yield O3(l=l, p=-1 * (-1) ** l)

    def continuous_generators(rep: "O3") -> np.ndarray:
        return SO3(l=rep.l).continuous_generators()

    def discrete_generators(rep: "O3") -> np.ndarray:
        return rep.p * np.eye(rep.dim)[None]

    def algebra(rep=None) -> np.ndarray:
        return SO3.algebra()
