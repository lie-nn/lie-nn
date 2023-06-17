import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from lie_nn import TabulatedIrrep
from lie_nn._src.lie_algebra_utils import RootsWeightsC, weyl_dim


def root_weights(rank):
    _, PositiveRoots, FundamentalWeights, _, _ = RootsWeightsC(rank)
    WeylVector = sum(PositiveRoots) / 2
    return PositiveRoots, FundamentalWeights, WeylVector


def weights(rep: "SPN") -> np.ndarray:
    fundamental_weights = root_weights(len(rep.S))[1]
    return sum(tuple(rep.S[i] * fundamental_weights[i] for i in range(len(rep.S))))


@dataclass(frozen=True)
class SPN(TabulatedIrrep):
    S: Tuple[int]  # List of weights of the representation with Dynkin labels

    @classmethod
    def from_string(cls, s: str) -> "SPN":
        # (4,3,2,1,0)
        m = re.match(r"\((\d+(?:,\d+)*)\)", s)
        return cls(S=tuple(map(int, m.group(1).split(","))))

    def __mul__(rep1: "SPN", rep2: "SPN") -> List["SPN"]:
        return NotImplementedError

    @classmethod
    def clebsch_gordan(cls, rep1: "SPN", rep2: "SPN", rep3: "SPN") -> np.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        return NotImplementedError

    def __lt__(rep1: "SPN", rep2: "SPN") -> bool:
        return rep1.S < rep2.S

    def __eq__(rep1: "SPN", rep2: "SPN") -> bool:
        return rep1.S == rep2.S

    @property
    def dim(rep: "SPN") -> int:
        positiveroots, weylvector = root_weights(len(rep.S))[0], root_weights(len(rep.S))[2]
        return int(weyl_dim(weights(rep), positiveroots, weylvector))

    def is_scalar(rep: "SPN") -> bool:
        """Equivalent to ``S=(0,...,0)``"""
        return all(s == 0 for s in rep.S)

    def discrete_generators(rep: "SPN") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SPN") -> np.ndarray:
        return NotImplementedError

    def algebra(rep: "SPN") -> np.ndarray:
        return NotImplementedError
