import numpy as np
from multipledispatch import dispatch

from .rep import GenericRep, Rep

# TODO(mario): Implement conjugate for Irreps


@dispatch(Rep)
def conjugate(rep: Rep) -> GenericRep:
    return GenericRep(
        A=rep.A,
        X=np.conjugate(rep.X),
        H=np.conjugate(rep.H),
    )
