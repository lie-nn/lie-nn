# import dataclasses

import numpy as np
from multipledispatch import dispatch

from . import GenericRep, Irrep, MulIrrep, ReducedRep, Rep


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
    return tensor_product(MulIrrep(mul=1, rep=irrep), rep)


@dispatch(Rep, Irrep)
def tensor_product(rep: Rep, irrep: Irrep) -> Rep:
    return tensor_product(rep, MulIrrep(mul=1, rep=irrep))


@dispatch(MulIrrep, Rep)
def tensor_product(mulirrep: MulIrrep, rep: Rep) -> Rep:
    return tensor_product(ReducedRep(A=mulirrep.algebra(), irreps=(mulirrep,), Q=None), rep)


@dispatch(Rep, MulIrrep)
def tensor_product(rep: Rep, mulirrep: MulIrrep) -> Rep:
    return tensor_product(rep, ReducedRep(A=mulirrep.algebra(), irreps=(mulirrep,), Q=None))


@dispatch(ReducedRep, ReducedRep)
def tensor_product(rep1: ReducedRep, rep2: ReducedRep) -> ReducedRep:
    pass


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
