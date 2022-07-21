import dataclasses

import numpy as np
from multipledispatch import dispatch

from . import GenericRep, Irrep, MulIrrep, ReducedRep, Rep


@dispatch(Rep, object)
def change_basis(rep: Rep, Q: np.ndarray) -> GenericRep:
    iQ = np.linalg.inv(Q)
    return GenericRep(
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
