import dataclasses

import numpy as np
from multipledispatch import dispatch
from typing import Tuple, Optional

from . import Irrep


class Rep:
    r"""Abstract representation of a Lie group."""

    @property
    def lie_dim(self) -> int:
        return self.algebra().shape[0]

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def algebra(self) -> np.ndarray:
        raise NotImplementedError

    def continuous_generators(self) -> np.ndarray:
        raise NotImplementedError

    def discrete_generators(self) -> np.ndarray:
        raise NotImplementedError


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
        raise NotImplementedError  # TODO direct sum

    def discrete_generators(self) -> np.ndarray:
        raise NotImplementedError  # TODO direct sum


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
        raise NotImplementedError  # TODO direct sum and change of basis

    def discrete_generators(self) -> np.ndarray:
        raise NotImplementedError  # TODO direct sum and change of basis


@dataclasses.dataclass
class UnknownRep(Rep):
    r"""Unknown representation"""
    A: np.ndarray
    X: np.ndarray
    H: np.ndarray

    @property
    def dim(self) -> int:
        return self.X.shape[1]

    def algebra(self) -> np.ndarray:
        return self.A

    def continuous_generators(self) -> np.ndarray:
        return self.X

    def discrete_generators(self) -> np.ndarray:
        return self.H


@dispatch(UnknownRep, object)
def change_basis(rep: UnknownRep, Q: np.ndarray) -> UnknownRep:
    iQ = np.linalg.inv(Q)
    return dataclasses.replace(
        rep,
        X=Q @ rep.X @ iQ,
        H=Q @ rep.H @ iQ,
    )


@dispatch(ReducedRep, object)
def change_basis(rep: ReducedRep, Q: np.ndarray) -> ReducedRep:
    Q = Q if rep.Q is None else Q @ rep.Q
    return dataclasses.replace(rep, Q=Q)


@dispatch(MulIrrep, object)
def change_basis(rep: MulIrrep, Q: np.ndarray) -> ReducedRep:
    return ReducedRep(
        A=rep.algebra(),
        irreps=(rep,),
        Q=Q,
    )


@dispatch(Irrep, object)
def change_basis(rep: Irrep, Q: np.ndarray) -> ReducedRep:
    return change_basis(MulIrrep(mul=1, rep=rep), Q)
