from multimethod import multimethod

import lie_nn as lie

import numpy as np


@multimethod
def multiply(mul: int, rep: lie.Rep) -> lie.Rep:
    if mul == 1:
        return rep

    return lie.MulRep(mul, rep, force=True)


@multimethod
def multiply(mul: int, mulrep: lie.MulRep) -> lie.Rep:  # noqa: F811
    return multiply(mul * mulrep.mul, mulrep.rep)


@multimethod
def multiply(mul: int, qrep: lie.QRep) -> lie.Rep:  # noqa: F811
    return lie.change_basis(lie.utils.direct_sum(*(qrep.Q,) * mul), multiply(mul, qrep.rep))


@multimethod
def multiply(mul: int, sumrep: lie.SumRep) -> lie.Rep:  # noqa: F811
    Q = np.zeros((mul, sumrep.dim, mul * sumrep.dim))
    k = 0
    j = 0
    for subrep in sumrep.reps:
        for u in range(mul):
            Q[u, k : k + subrep.dim, j : j + subrep.dim] = np.eye(subrep.dim)
            j += subrep.dim
        k += subrep.dim
    Q = Q.reshape(mul * sumrep.dim, mul * sumrep.dim)
    return lie.change_basis(Q, lie.direct_sum(*[multiply(mul, subrep) for subrep in sumrep.reps]))
