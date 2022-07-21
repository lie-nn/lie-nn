# import dataclasses

from lie_nn.util import block_diagonal
import numpy as np
from multipledispatch import dispatch

from .irreps import GenericRep, Irrep, MulIrrep, ReducedRep, Rep


@dispatch(Rep, Rep)
def tensor_product(rep1: Rep, rep2: Rep) -> GenericRep:
    assert np.allclose(rep1.algebra(), rep2.algebra())  # same lie algebra
    X1, H1 = rep1.continuous_generators(), rep1.discrete_generators()
    X2, H2 = rep2.continuous_generators(), rep2.discrete_generators()
    assert H1.shape[0] == H2.shape[0]  # same discrete dimension
    d = rep1.dim * rep2.dim
    return GenericRep(
        A=rep1.algebra(),
        X=np.einsum("aij,akl->aikjl", X1, X2).reshape(X1.shape[0], d, d),
        H=np.einsum("aij,akl->aikjl", H1, H2).reshape(H1.shape[0], d, d),
    )


@dispatch(Irrep, Rep)
def tensor_product(irrep: Irrep, rep: Rep) -> Rep:
    assert np.allclose(irrep.algebra(), rep.algebra())  # same lie algebra
    return tensor_product(MulIrrep(mul=1, rep=irrep), rep)


@dispatch(Rep, Irrep)
def tensor_product(rep: Rep, irrep: Irrep) -> Rep:
    assert np.allclose(rep.algebra(), irrep.algebra())  # same lie algebra
    return tensor_product(rep, MulIrrep(mul=1, rep=irrep))


@dispatch(MulIrrep, Rep)
def tensor_product(mulirrep: MulIrrep, rep: Rep) -> Rep:
    assert np.allclose(mulirrep.rep.algebra(), rep.algebra())  # same lie algebra
    return tensor_product(ReducedRep(A=mulirrep.algebra(), irreps=(mulirrep,), Q=None), rep)


@dispatch(Rep, MulIrrep)
def tensor_product(rep: Rep, mulirrep: MulIrrep) -> Rep:
    assert np.allclose(rep.algebra(), mulirrep.rep.algebra())  # same lie algebra
    return tensor_product(rep, ReducedRep(A=mulirrep.algebra(), irreps=(mulirrep,), Q=None))


@dispatch(Irrep, Irrep)
def tensor_product(irrep1: Irrep, irrep2: Irrep) -> ReducedRep:
    assert np.allclose(irrep1.algebra(), irrep2.algebra())  # same lie algebra
    CG_list = []
    irreps_list = []
    for ir_out in irrep1 * irrep2:
        CG = np.moveaxis(irrep1.clebsch_gordan(irrep1, irrep2, ir_out), 0, -2)
        mul = CG.shape[-2]
        CG = CG.reshape(CG.shape[0] * CG.shape[1], CG.shape[-2] * CG.shape[-1])
        CG_list.append(CG)
        irreps_list.append(MulIrrep(mul=mul, rep=ir_out))
    CG = np.concatenate(CG_list, axis=-1)
    return ReducedRep(A=irrep1.algebra(), irreps=tuple(irreps_list), Q=CG)


@dispatch(MulIrrep, MulIrrep)
def tensor_product(mulirrep1: MulIrrep, mulirrep2: MulIrrep) -> ReducedRep:
    assert np.allclose(mulirrep1.algebra(), mulirrep2.algebra())  # same lie algebra
    tp_irreps = tensor_product(mulirrep1.rep, mulirrep2.rep)
    Q = tp_irreps.Q
    tp_irreps.mul = mulirrep1.mul * mulirrep2.mul
    Q_out = np.einsum('ue,vs,ijk->uivjesk', np.eye(mulirrep1.mul), np.eye(mulirrep2.mul), Q)
    return ReducedRep(A=mulirrep1.algebra(), irreps=tuple(tp_irreps.irreps), Q=Q_out)


@dispatch(ReducedRep, ReducedRep)
def tensor_product(rep1: ReducedRep, rep2: ReducedRep) -> ReducedRep:
    Q_tp = np.einsum('ij,kl->ijkl', rep1.Q, rep2.Q)
    CG_list = []
    reducedrep_list = []
    for mulirrep1 in rep1.irreps:
        for mulirrep2 in enumerate(rep2.irreps):
            reducedrep = tensor_product(mulirrep1, mulirrep2)
            reducedrep_list.append(reducedrep)
            CG_list.append(reducedrep.Q)
    CG = np.concatenate(CG_list, axis=-1)
    Q = Q_tp @ CG
    return ReducedRep(A=rep1.A, irreps=tuple(reducedrep_list), Q=Q)


@dispatch(Rep, int)
def tensor_power(rep: Rep, n: int) -> GenericRep:
    X, H = rep.continuous_generators(), rep.discrete_generators()
    result = GenericRep(
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
