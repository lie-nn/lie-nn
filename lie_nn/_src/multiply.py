from multipledispatch import dispatch

from .rep import Rep, MulRep, QRep, SumRep
from .util import direct_sum as ds
from .change_basis import change_basis
from .direct_sum import direct_sum
import numpy as np


@dispatch(int, Rep)
def multiply(mul: int, rep: Rep) -> Rep:
    if mul == 1:
        return rep

    return MulRep(mul, rep, force=True)


@dispatch(int, MulRep)
def multiply(mul: int, mulrep: MulRep) -> MulRep:  # noqa: F811
    return multiply(mul * mulrep.mul, mulrep.rep)


@dispatch(int, QRep)
def multiply(mul: int, qrep: QRep) -> Rep:  # noqa: F811
    return change_basis(ds(*(qrep.Q,) * mul), multiply(mul, qrep.rep))


@dispatch(int, SumRep)
def multiply(mul: int, sumrep: SumRep) -> Rep:  # noqa: F811
    Q = np.zeros((mul, sumrep.dim, mul * sumrep.dim))
    k = 0
    j = 0
    for subrep in sumrep.reps:
        for u in range(mul):
            Q[u, k : k + subrep.dim, j : j + subrep.dim] = np.eye(subrep.dim)
            j += subrep.dim
        k += subrep.dim
    Q = Q.reshape(mul * sumrep.dim, mul * sumrep.dim)
    return change_basis(Q, direct_sum(*[multiply(mul, subrep) for subrep in sumrep.reps]))
