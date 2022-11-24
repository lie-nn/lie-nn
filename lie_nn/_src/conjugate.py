import numpy as np
from multipledispatch import dispatch

from .rep import GenericRep, Rep

# TODO(mario): Implement conjugate for Irreps


@dispatch(Rep)
def conjugate(rep: Rep) -> GenericRep:
    return GenericRep(A=rep.algebra(), X=np.conjugate(rep.continuous_generators()), H=np.conjugate(rep.discrete_generators()),)

