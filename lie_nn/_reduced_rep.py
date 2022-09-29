import dataclasses
from typing import Optional, Tuple, Union

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

    def create_trivial(self) -> Irrep:
        return self.rep.create_trivial()

    def __repr__(self) -> str:
        return f"{self.mul}x{self.rep}"


@dataclasses.dataclass
class ReducedRep(Rep):
    r"""Representation of the form

    .. math::
        Q (\osum_i m_i \rho_i ) Q^{-1}
    """
    A: np.ndarray
    irreps: Tuple[MulIrrep, ...]
    Q: Optional[np.ndarray]  # change of basis matrix

    @classmethod
    def from_irreps(cls, mul_irreps: Tuple[Union[Irrep, Tuple[int, Irrep], MulIrrep], ...]) -> "ReducedRep":
        A = None
        irreps = []

        for mul_ir in mul_irreps:
            if isinstance(mul_ir, tuple):
                mul_ir = MulIrrep(mul=mul_ir[0], rep=mul_ir[1])
            elif isinstance(mul_ir, Irrep):
                mul_ir = MulIrrep(mul=1, rep=mul_ir)
            elif isinstance(mul_ir, MulIrrep):
                pass

            assert mul_ir.mul >= 0
            assert isinstance(mul_ir.rep, Irrep)

            irreps += [mul_ir]

            if A is None:
                A = mul_ir.algebra()
            else:
                assert np.allclose(A, mul_ir.algebra())

        return cls(A=A, irreps=irreps, Q=None)

    @property
    def dim(self) -> int:
        return sum(irrep.dim for irrep in self.irreps)

    def algebra(self) -> np.ndarray:
        return self.A

    def continuous_generators(self) -> np.ndarray:
        Xs = []
        for i in range(self.lie_dim):
            X = direct_sum(*[mul_ir.continuous_generators()[i] for mul_ir in self.irreps])
            if self.Q is not None:
                X = self.Q @ X @ np.linalg.inv(self.Q)
            Xs += [X]
        return np.stack(Xs)

    def discrete_generators(self) -> np.ndarray:
        n = self.irreps[0].discrete_generators().shape[0]  # TODO: support empty irreps
        if n == 0:
            return np.empty((0, self.dim, self.dim))
        Xs = []
        for i in range(n):
            X = direct_sum(*[mul_ir.discrete_generators()[i] for mul_ir in self.irreps])
            if self.Q is not None:
                X = self.Q @ X @ np.linalg.inv(self.Q)
            Xs += [X]
        return np.stack(Xs)

    def create_trivial(self) -> Irrep:
        return self.irreps[0].create_trivial()

    def __repr__(self) -> str:
        r = " + ".join(repr(mul_ir) for mul_ir in self.irreps)
        if self.Q is not None:
            r = f"Q ({r}) Q^{-1}"
        return r
