# Group direct product

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from multipledispatch import dispatch

from .irrep import Irrep
from .rep import GenericRep, Rep


def _get_dtype(*args):
    x = 0.0
    for arg in args:
        x = x + np.array(0.0, dtype=arg.dtype)
    return x.dtype


@dispatch(Rep, Rep)
def group_product(rep1: Rep, rep2: Rep) -> GenericRep:
    A1 = rep1.A
    A2 = rep2.A
    lie_dim = rep1.lie_dim + rep2.lie_dim
    A = np.zeros((lie_dim, lie_dim, lie_dim), dtype=_get_dtype(A1, A2))
    A[: rep1.lie_dim, : rep1.lie_dim, : rep1.lie_dim] = A1
    A[rep1.lie_dim :, rep1.lie_dim :, rep1.lie_dim :] = A2

    dim = rep1.dim * rep2.dim
    I1 = np.eye(rep1.dim)
    I2 = np.eye(rep2.dim)

    X1 = rep1.X
    X2 = rep2.X
    X = np.zeros((lie_dim, dim, dim), dtype=_get_dtype(X1, X2))
    X[: rep1.lie_dim] = np.einsum("aij,kl->aikjl", X1, I2).reshape(rep1.lie_dim, dim, dim)
    X[rep1.lie_dim :] = np.einsum("ij,akl->aikjl", I1, X2).reshape(rep2.lie_dim, dim, dim)

    H1 = rep1.H
    H2 = rep2.H
    H = np.zeros((H1.shape[0] + H2.shape[0], dim, dim), dtype=_get_dtype(H1, H2))
    H[: H1.shape[0]] = np.einsum("aij,kl->aikjl", H1, I2).reshape(H1.shape[0], dim, dim)
    H[H1.shape[0] :] = np.einsum("ij,akl->aikjl", I1, H2).reshape(H2.shape[0], dim, dim)

    return GenericRep(A=A, X=X, H=H)


@dispatch(Rep, Rep, Rep)
def group_product(rep1: Rep, rep2: Rep, rep3: Rep) -> GenericRep:
    return group_product(group_product(rep1, rep2), rep3)


@dataclass(frozen=True)
class IrrepProduct(Irrep):
    rep1: Irrep
    rep2: Irrep

    @classmethod
    def from_string(cls, s: str) -> "IrrepProduct":
        raise NotImplementedError

    def __mul__(rep1: "IrrepProduct", rep2: "IrrepProduct") -> Iterator["IrrepProduct"]:
        assert isinstance(rep2, IrrepProduct)
        for rep3 in rep1.rep1 * rep2.rep1:
            for rep4 in rep1.rep2 * rep2.rep2:
                yield IrrepProduct(rep3, rep4)

    @classmethod
    def clebsch_gordan(cls, rep1: "IrrepProduct", rep2: "IrrepProduct", rep3: "IrrepProduct") -> np.ndarray:
        C1 = rep1.rep1.clebsch_gordan(rep1.rep1, rep2.rep1, rep3.rep1)  # [n_sol1, d1_1, d2_1, d3_1]
        C2 = rep1.rep2.clebsch_gordan(rep1.rep2, rep2.rep2, rep3.rep2)  # [n_sol2, d1_2, d2_2, d3_2]
        C = np.einsum("aikm,bjln->abijklmn", C1, C2).reshape(len(C1) * len(C2), rep1.dim, rep2.dim, rep3.dim)
        return C

    @property
    def dim(rep: "IrrepProduct") -> int:
        return rep.rep1.dim * rep.rep2.dim

    def __lt__(rep1: "IrrepProduct", rep2: "IrrepProduct") -> bool:
        return (rep1.rep1, rep1.rep2) < (rep2.rep1, rep2.rep2)

    @classmethod
    def iterator(cls) -> Iterator["IrrepProduct"]:
        raise NotImplementedError

    def continuous_generators(rep: "IrrepProduct") -> np.ndarray:
        X1 = rep.rep1.X
        X2 = rep.rep2.X
        I1 = np.eye(rep.rep1.dim)
        I2 = np.eye(rep.rep2.dim)
        X = np.zeros((rep.lie_dim, rep.dim, rep.dim), dtype=_get_dtype(X1, X2))
        X[: rep.rep1.lie_dim] = np.einsum("aij,kl->aikjl", X1, I2).reshape(rep.rep1.lie_dim, rep.dim, rep.dim)
        X[rep.rep1.lie_dim :] = np.einsum("ij,akl->aikjl", I1, X2).reshape(rep.rep2.lie_dim, rep.dim, rep.dim)
        return X

    def discrete_generators(rep: "IrrepProduct") -> np.ndarray:
        H1 = rep.rep1.H
        H2 = rep.rep2.H
        I1 = np.eye(rep.rep1.dim)
        I2 = np.eye(rep.rep2.dim)
        H = np.zeros((H1.shape[0] + H2.shape[0], rep.dim, rep.dim), dtype=_get_dtype(H1, H2))
        H[: H1.shape[0]] = np.einsum("aij,kl->aikjl", H1, I2).reshape(H1.shape[0], rep.dim, rep.dim)
        H[H1.shape[0] :] = np.einsum("ij,akl->aikjl", I1, H2).reshape(H2.shape[0], rep.dim, rep.dim)
        return H

    def algebra(rep: "IrrepProduct") -> np.ndarray:
        A1 = rep.rep1.A
        A2 = rep.rep2.A
        lie_dim = rep.rep1.lie_dim + rep.rep2.lie_dim
        A = np.zeros((lie_dim, lie_dim, lie_dim), dtype=_get_dtype(A1, A2))
        A[: rep.rep1.lie_dim, : rep.rep1.lie_dim, : rep.rep1.lie_dim] = A1
        A[rep.rep1.lie_dim :, rep.rep1.lie_dim :, rep.rep1.lie_dim :] = A2
        return A


@dispatch(Irrep, Irrep)
def group_product(rep1: Irrep, rep2: Irrep) -> Irrep:
    return IrrepProduct(rep1, rep2)


@dispatch(Irrep, Irrep, Irrep)
def group_product(rep1: Irrep, rep2: Irrep, rep3: Irrep) -> Irrep:
    return IrrepProduct(IrrepProduct(rep1, rep2), rep3)