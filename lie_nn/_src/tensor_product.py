import numpy as np
from multipledispatch import dispatch

import lie_nn as lie

from .irrep import TabulatedIrrep
from .reduced_rep import MulIrrep, ReducedRep
from .rep import GenericRep, Rep


@dispatch(Rep, Rep)
def tensor_product(rep1: Rep, rep2: Rep) -> GenericRep:
    assert np.allclose(rep1.A, rep2.A)  # same lie algebra
    X1, H1, I1 = rep1.X, rep1.H, np.eye(rep1.dim)
    X2, H2, I2 = rep2.X, rep2.H, np.eye(rep2.dim)
    assert H1.shape[0] == H2.shape[0]  # same discrete dimension
    d = rep1.dim * rep2.dim
    return GenericRep(
        A=rep1.A,
        X=(np.einsum("aij,kl->aikjl", X1, I2) + np.einsum("ij,akl->aikjl", I1, X2)).reshape(
            X1.shape[0], d, d
        ),
        H=np.einsum("aij,akl->aikjl", H1, H2).reshape(H1.shape[0], d, d),
    )


@dispatch(TabulatedIrrep, TabulatedIrrep)
def tensor_product(irrep1: TabulatedIrrep, irrep2: TabulatedIrrep) -> ReducedRep:  # noqa: F811
    assert np.allclose(irrep1.A, irrep2.A)  # same lie algebra
    CG_list = []
    irreps_list = []
    for ir_out in irrep1 * irrep2:
        CG = np.moveaxis(lie.clebsch_gordan(irrep1, irrep2, ir_out), 3, 1)  # [sol, ir3, ir1, ir2]
        mul = CG.shape[0]
        CG = CG.reshape(CG.shape[0] * CG.shape[1], CG.shape[2] * CG.shape[3])
        CG_list.append(CG)
        irreps_list.append(MulIrrep(mul=mul, rep=ir_out))
    CG = np.concatenate(CG_list, axis=0)
    Q = np.linalg.inv(CG)
    return ReducedRep(A=irrep1.A, irreps=tuple(irreps_list), Q=Q)


@dispatch(MulIrrep, MulIrrep)
def tensor_product(mulirrep1: MulIrrep, mulirrep2: MulIrrep) -> ReducedRep:  # noqa: F811
    assert np.allclose(mulirrep1.A, mulirrep2.A)  # same lie algebra
    m1, m2 = mulirrep1.mul, mulirrep2.mul
    tp_irreps = tensor_product(mulirrep1.rep, mulirrep2.rep)
    Q = tp_irreps.Q
    irreps = tp_irreps.irreps
    Q_out = []
    irreps_out = []
    s = 0
    for mul_ir in irreps:
        irreps_out.append(MulIrrep(mul=m1 * m2 * mul_ir.mul, rep=mul_ir.rep))
        q = Q[:, s : s + mul_ir.dim].reshape(
            mulirrep1.rep.dim, mulirrep2.rep.dim, mul_ir.mul, mul_ir.rep.dim
        )
        q = np.einsum("ijsk,ur,vt->uivjrtsk", q, np.eye(m1), np.eye(m2))
        q = q.reshape(
            q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3],
            q.shape[4] * q.shape[5] * q.shape[6] * q.shape[7],
        )
        Q_out.append(q)
        s += mul_ir.dim
    Q_out = np.concatenate(Q_out, axis=-1)
    return ReducedRep(A=mulirrep1.A, irreps=tuple(irreps_out), Q=Q_out)


@dispatch(ReducedRep, ReducedRep)
def tensor_product(rep1: ReducedRep, rep2: ReducedRep) -> ReducedRep:  # noqa: F811
    q1 = np.eye(rep1.dim) if rep1.Q is None else rep1.Q
    q2 = np.eye(rep2.dim) if rep2.Q is None else rep2.Q
    Q_tp = np.einsum("ij,kl->ikjl", q1, q2).reshape(rep1.dim * rep2.dim, rep1.dim * rep2.dim)
    mulir_list = []

    Q = np.zeros((rep1.dim, rep2.dim, rep1.dim * rep2.dim), dtype=np.complex128)
    k = 0
    i = 0
    for mulirrep1 in rep1.irreps:
        j = 0
        for mulirrep2 in rep2.irreps:
            reducedrep = tensor_product(mulirrep1, mulirrep2)
            mulir_list += reducedrep.irreps
            q = reducedrep.Q.reshape(mulirrep1.dim, mulirrep2.dim, reducedrep.dim)
            Q[i : i + mulirrep1.dim, j : j + mulirrep2.dim, k : k + reducedrep.dim] = q
            k += reducedrep.dim
            j += mulirrep2.dim
        i += mulirrep1.dim
    assert k == rep1.dim * rep2.dim
    Q = Q.reshape(rep1.dim * rep2.dim, rep1.dim * rep2.dim)
    Q = Q_tp @ Q

    if np.allclose(Q.imag, 0):
        Q = Q.real
    if np.allclose(Q, np.eye(Q.shape[0])):
        Q = None
    return ReducedRep(A=rep1.A, irreps=tuple(mulir_list), Q=Q)


@dispatch(MulIrrep, TabulatedIrrep)
def tensor_product(mulirrep1: MulIrrep, irrep2: TabulatedIrrep) -> ReducedRep:  # noqa: F811
    return tensor_product(mulirrep1, MulIrrep(mul=1, rep=irrep2))


@dispatch(TabulatedIrrep, MulIrrep)
def tensor_product(irrep1: TabulatedIrrep, mulirrep2: MulIrrep) -> ReducedRep:  # noqa: F811
    return tensor_product(MulIrrep(mul=1, rep=irrep1), mulirrep2)


@dispatch(MulIrrep, ReducedRep)
def tensor_product(mulirrep1: MulIrrep, rep2: ReducedRep) -> ReducedRep:  # noqa: F811
    return tensor_product(ReducedRep(A=mulirrep1.A, irreps=(mulirrep1,), Q=None), rep2)


@dispatch(ReducedRep, MulIrrep)
def tensor_product(rep1: ReducedRep, mulirrep2: MulIrrep) -> ReducedRep:  # noqa: F811
    return tensor_product(ReducedRep(A=mulirrep2.A, irreps=(mulirrep2,), Q=None), rep1)


@dispatch(ReducedRep, TabulatedIrrep)
def tensor_product(rep1: ReducedRep, irrep2: TabulatedIrrep) -> ReducedRep:  # noqa: F811
    return tensor_product(rep1, MulIrrep(mul=1, rep=irrep2))


@dispatch(TabulatedIrrep, ReducedRep)
def tensor_product(irrep1: TabulatedIrrep, rep2: ReducedRep) -> ReducedRep:  # noqa: F811
    return tensor_product(MulIrrep(mul=1, rep=irrep1), rep2)


# @dispatch(Rep, int)
def tensor_power(rep: Rep, n: int) -> Rep:
    result = rep.create_trivial()

    while True:
        if n & 1:
            result = tensor_product(rep, result)
        n >>= 1

        if n == 0:
            return result
        rep = tensor_product(rep, rep)


# @dispatch(ReducedRep, int)
# def tensor_power(rep: ReducedRep, n: int) -> ReducedRep:
#     # TODO reduce into irreps and wrap with the change of basis that
#       maps to the usual tensor product
#     # TODO as well reduce into irreps of S_n
#     # and diagonalize irreps of S_n in the same basis that diagonalizes
#       irreps of S_{n-1} (unclear how to do this)
#     raise NotImplementedError
