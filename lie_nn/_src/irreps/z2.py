from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..irrep import TabulatedIrrep


@dataclass(frozen=True)
class Z2(TabulatedIrrep):
    p: int  # 1 or -1

    @classmethod
    def from_string(cls, s: str) -> "Z2":
        s = s.strip()
        p = {"e": 1, "o": -1}[s]
        return cls(p=p)

    def __mul__(rep1: "Z2", rep2: "Z2") -> Iterator["Z2"]:
        assert isinstance(rep2, Z2)
        p = rep1.p * rep2.p
        return [Z2(p=p)]

    @classmethod
    def clebsch_gordan(cls, rep1: "Z2", rep2: "Z2", rep3: "Z2") -> np.ndarray:
        if rep1.p * rep2.p == rep3.p:
            return np.ones((1, rep1.dim, rep2.dim, rep3.dim))
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: "Z2") -> int:
        return 1

    def __lt__(rep1: "Z2", rep2: "Z2") -> bool:
        # 1, -1
        return rep1.p > rep2.p

    @classmethod
    def iterator(cls) -> Iterator["Z2"]:
        yield Z2(p=1)
        yield Z2(p=-1)

    def continuous_generators(rep: "Z2") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def discrete_generators(rep: "Z2") -> np.ndarray:
        return rep.p * np.eye(rep.dim)[None]

    def algebra(rep=None) -> np.ndarray:
        return np.zeros((0, 0, 0))
