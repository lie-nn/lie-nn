import numpy as np
from multipledispatch import dispatch

from .rep import GenericRep, Rep
from .util import direct_sum as _direct_sum


@dispatch(Rep, Rep)
def direct_sum(rep1: Rep, rep2: Rep) -> GenericRep:
    assert np.allclose(rep1.A, rep2.A)  # same lie algebra
    assert rep1.H.shape[0] == rep2.H.shape[0]  # same discrete dimension
    return GenericRep(
        A=rep1.A,
        X=_direct_sum(rep1.X, rep2.X),
        H=_direct_sum(rep1.H, rep2.H),
    )
