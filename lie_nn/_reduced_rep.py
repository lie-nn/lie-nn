import dataclasses
from typing import Optional, Tuple

import numpy as np

from . import Irrep, Rep
from .util import direct_sum


@dataclasses.dataclass
class MulIrrep(Rep):
    mul: int
    rep: Irrep

    @property
    def dim(self) -> int:
        return self.mul * self.rep.dim

    def algebra(self) -> np.ndarray:
        return self.rep.algebra()

    def continuous_generators(self) -> np.ndarray:
        X = self.rep.continuous_generators()
        return np.stack([direct_sum(*[x for _ in range(self.mul)]) for x in X], axis=0)

    def discrete_generators(self) -> np.ndarray:
        H = self.rep.discrete_generators()
        if H.shape[0] == 0:
            return np.empty((0, self.dim, self.dim))
        return np.stack([direct_sum(*[x for _ in range(self.mul)]) for x in H], axis=0)


@dataclasses.dataclass
class ReducedRep(Rep):
    r"""Representation of the form

    .. math::
        Q (\osum_i m_i \rho_i ) Q^{-1}
    """
    A: np.ndarray
    irreps: Tuple[MulIrrep, ...]
    Q: Optional[np.ndarray]  # change of basis matrix

    @property
    def dim(self) -> int:
        return sum(irrep.dim for irrep in self.irreps)

    def algebra(self) -> np.ndarray:
        return self.A

    def continuous_generators(self) -> np.ndarray:
        Xs = []
        for i in range(self.lie_dim):
            X = direct_sum(*[mulir.continuous_generators()[i] for mulir in self.irreps])
            X = self.Q @ X @ np.linalg.inv(self.Q)
            Xs += [X]
        return np.stack(Xs)

    def discrete_generators(self) -> np.ndarray:
        n = self.irreps[0].discrete_generators().shape[0]  # TODO: support empty irreps
        if n == 0:
            return np.empty((0, self.dim, self.dim))
        Xs = []
        for i in range(n):
            X = direct_sum(*[irrep.discrete_generators()[i] for irrep in self.irreps])
            X = self.Q @ X @ np.linalg.inv(self.Q)
            Xs += [X]
        return np.stack(Xs)
