import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import lie_nn as lie
from lie_nn import TabulatedIrrep
from lie_nn._src.lie_algebra_utils import RootsWeightsB, RootsWeightsD, weyl_dim


def root_weights(rank):
    if rank % 2 == 0:
        _, PositiveRoots, FundamentalWeights, _, _ = RootsWeightsD(rank)
    else:
        _, PositiveRoots, FundamentalWeights, _, _ = RootsWeightsB(rank)
    WeylVector = sum(PositiveRoots) / 2
    return PositiveRoots, FundamentalWeights, WeylVector


def weights(rep: "SON") -> np.ndarray:
    fundamental_weights = root_weights(len(rep.S))[1]
    return sum(tuple(rep.S[i] * fundamental_weights[i] for i in range(len(rep.S))))


@dataclass(frozen=True)
class SON(TabulatedIrrep):
    S: Tuple[int]  # List of weights of the representation with Dynkin labels

    @classmethod
    def from_string(cls, s: str) -> "SON":
        # (4,3,2,1,0)
        m = re.match(r"\((\d+(?:,\d+)*)\)", s)
        return cls(S=tuple(map(int, m.group(1).split(","))))

    def __mul__(rep1: "SON", rep2: "SON") -> List["SON"]:
        return NotImplementedError

    @classmethod
    def clebsch_gordan(cls, rep1: "SON", rep2: "SON", rep3: "SON") -> np.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        return NotImplementedError

    def __lt__(rep1: "SON", rep2: "SON") -> bool:
        return rep1.S < rep2.S

    def __eq__(rep1: "SON", rep2: "SON") -> bool:
        return rep1.S == rep2.S

    @property
    def dim(rep: "SON") -> int:
        positiveroots, weylvector = root_weights(len(rep.S))[0], root_weights(len(rep.S))[2]
        return int(weyl_dim(weights(rep), positiveroots, weylvector))

    def is_scalar(rep: "SON") -> bool:
        """Equivalent to ``S=(0,...,0)``"""
        return all(s == 0 for s in rep.S)

    def discrete_generators(rep: "SON") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SON") -> np.ndarray:
        return NotImplementedError

    def algebra(rep: "SON") -> np.ndarray:
        return NotImplementedError


def E(i, j, rank):
    """Returns the matrix with a 1 in the (i,j) entry and 0s elsewhere"""
    E = np.zeros((rank, rank))
    E[i, j] = 1
    return E


class SON_adjoint(lie.Rep):
    """Adjoint representation of SON" from Fulton-Harris"""

    def __init__(self, n) -> None:
        super().__init__()
        assert n >= 3
        self.n = n

    def algebra(self) -> np.ndarray:
        return NotImplementedError

    def continuous_generators(self) -> np.ndarray:
        if self.n % 2 == 0:
            n2 = self.n // 2
            Hs = np.stack([E(i, i, self.n) - E(i + n2, i + n2, self.n) for i in range(n2)])
            Xs = np.stack(
                [
                    E(i, j, self.n) - E(n2 + j, n2 + i, self.n)
                    for i in range(n2)
                    for j in range(n2)
                    if i != j
                ]
            )
            Ys = np.stack(
                [
                    E(i, n2 + j, self.n) - E(j, n2 + i, self.n)
                    for i in range(n2)
                    for j in range(n2)
                    if i != j
                ]
            )
            Zs = np.stack(
                [
                    E(n2 + i, j, self.n) - E(n2 + j, i, self.n)
                    for i in range(n2)
                    for j in range(n2)
                    if i != j
                ]
            )
            return np.concatenate([Hs, Xs, Ys, Zs])
        else:
            n2 = (self.n - 1) // 2
            Hs = np.stack([E(i, i, self.n) - E(i + n2, i + n2, self.n) for i in range(n2)])
            Xs = np.stack(
                [
                    E(i, j, self.n) - E(n2 + j, n2 + i, self.n)
                    for i in range(n2)
                    for j in range(n2)
                    if i != j
                ]
            )
            Ys = np.stack(
                [
                    E(i, n2 + j, self.n) - E(j, n2 + i, self.n)
                    for i in range(n2)
                    for j in range(n2)
                    if i != j
                ]
            )
            Zs = np.stack(
                [
                    E(n2 + i, j, self.n) - E(n2 + j, i, self.n)
                    for i in range(n2)
                    for j in range(n2)
                    if i != j
                ]
            )
            Us = np.stack(
                [E(i, self.n - 1, self.n) - E(self.n - 1, i, self.n) for i in range(self.n)]
            )
            Vs = np.stack(
                [E(n2 + i, self.n - 1, self.n) - E(self.n - 1, i, self.n) for i in range(n2)]
            )
            return np.concatenate([Hs, Xs, Ys, Zs, Us, Vs])

    def discrete_generators(self) -> np.ndarray:
        return np.zeros((0, self.n, self.n))

    def create_trivial(self) -> lie.GenericRep:
        return lie.GenericRep(
            A=self.A, X=np.ones((0, self.n, self.n)), H=np.zeros((0, self.n, self.n))
        )
