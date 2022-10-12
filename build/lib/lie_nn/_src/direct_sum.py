import numpy as np
from multipledispatch import dispatch

from .rep import GenericRep, Rep
from .util import direct_sum as _direct_sum


@dispatch(Rep, Rep)
def direct_sum(rep1: Rep, rep2: Rep) -> GenericRep:
    assert np.allclose(rep1.algebra(), rep2.algebra())  # same lie algebra
    X1, H1 = rep1.continuous_generators(), rep1.discrete_generators()
    X2, H2 = rep2.continuous_generators(), rep2.discrete_generators()
    assert H1.shape[0] == H2.shape[0]  # same discrete dimension
    return GenericRep(
        A=rep1.algebra(),
        X=_direct_sum(X1, X2),
        H=_direct_sum(H1, H2),
    )
