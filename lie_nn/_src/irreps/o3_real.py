import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..irrep import Irrep
from .so3_real import SO3Rep


@dataclass(frozen=True)
class O3Rep(Irrep):
    l: int  # non-negative integer
    p: int  # 1 or -1

    @classmethod
    def from_string(cls, s: str) -> "O3Rep":
        s = s.strip()
        l = int(s[:-1])
        p = {"e": 1, "o": -1}[s[-1]]
        return cls(l=l, p=p)

    def __mul__(rep1: "O3Rep", rep2: "O3Rep") -> Iterator["O3Rep"]:
        assert isinstance(rep2, O3Rep)
        p = rep1.p * rep2.p
        return [O3Rep(l=l, p=p) for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, rep1: "O3Rep", rep2: "O3Rep", rep3: "O3Rep") -> np.ndarray:
        if rep1.p * rep2.p == rep3.p:
            return SO3Rep.clebsch_gordan(rep1, rep2, rep3)
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: "O3Rep") -> int:
        return 2 * rep.l + 1

    def __lt__(rep1: "O3Rep", rep2: "O3Rep") -> bool:
        # scalar, speudo-scalar, vector, pseudo-vector, tensor, pseudo-tensor, ...
        return (rep1.l, -((-1) ** rep1.l) * rep1.p) < (rep2.l, -((-1) ** rep2.l) * rep2.p)

    @classmethod
    def iterator(cls) -> Iterator["O3Rep"]:
        for l in itertools.count(0):
            yield O3Rep(l=l, p=1 * (-1) ** l)
            yield O3Rep(l=l, p=-1 * (-1) ** l)

    def continuous_generators(rep: "O3Rep") -> np.ndarray:
        return SO3Rep(l=rep.l).continuous_generators()

    def discrete_generators(rep: "O3Rep") -> np.ndarray:
        return rep.p * np.eye(rep.dim)[None]

    def algebra(rep=None) -> np.ndarray:
        return SO3Rep.algebra()