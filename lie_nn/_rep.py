import dataclasses

import numpy as np
from multipledispatch import dispatch
from typing import Tuple, Optional

from . import Irrep


class Rep:
    r"""Abstract representation of a Lie group."""

    @property
    def lie_dim(self) -> int:
        A = self.algebra()
        d = A.shape[0]
        assert A.shape == (d, d, d)
        # X = self.continuous_generators()
        # assert X.shape[0] == d
        return d

    @property
    def dim(self) -> int:
        X = self.continuous_generators()
        d = X.shape[1]
        # H = self.discrete_generators()
        # assert X.shape[1:] == (d, d)
        # assert H.shape[1:] == (d, d)
        return d

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

    def algebra(self) -> np.ndarray:
        return self.A

    def continuous_generators(self) -> np.ndarray:
        return self.X

    def discrete_generators(self) -> np.ndarray:
        return self.H


@dispatch(Rep, object)
def change_basis(rep: Rep, Q: np.ndarray) -> UnknownRep:
    iQ = np.linalg.inv(Q)
    return UnknownRep(
        A=rep.algebra(),
        X=Q @ rep.continuous_generators() @ iQ,
        H=Q @ rep.discrete_generators() @ iQ,
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


@dispatch(Rep, Rep)
def tensor_product(rep1: Rep, rep2: Rep) -> UnknownRep:
    assert np.allclose(rep1.algebra(), rep2.algebra())  # same lie algebra
    X1, H1 = rep1.continuous_generators(), rep1.discrete_generators()
    X2, H2 = rep2.continuous_generators(), rep2.discrete_generators()
    assert H1.shape[0] == H2.shape[0]  # same discrete dimension
    d = rep1.dim * rep2.dim
    return UnknownRep(
        A=rep1.algebra(),
        X=np.einsum("aij,akl->aikjl", X1, X2).reshape(X1.shape[0], d, d),
        H=np.einsum("aij,akl->aikjl", H1, H2).reshape(H1.shape[0], d, d),
    )


@dispatch(ReducedRep, ReducedRep)
def tensor_product(rep1: ReducedRep, rep2: ReducedRep) -> ReducedRep:
    raise NotImplementedError


@dispatch(Rep, int)
def tensor_power(rep: Rep, n: int) -> UnknownRep:
    X, H = rep.continuous_generators(), rep.discrete_generators()
    result = UnknownRep(
        A=rep.algebra(),
        X=np.ones((X.shape[0], 1, 1)),
        H=np.ones((H.shape[0], 1, 1)),
    )

    while True:
        if n & 1:
            result = tensor_product(rep, result)
        n >>= 1

        if n == 0:
            return result

        rep = tensor_product(rep, rep)


@dispatch(ReducedRep, int)
def tensor_power(rep: ReducedRep, n: int) -> ReducedRep:
    # TODO reduce into irreps and wrap with the change of basis that maps to the usual tensor product
    # TODO as well reduce into irreps of S_n
    # and diagonalize irreps of S_n in the same basis that diagonalizes irreps of S_{n-1} (unclear how to do this)
    raise NotImplementedError
