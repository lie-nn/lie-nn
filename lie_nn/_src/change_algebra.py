import numpy as np
from multimethod import multimethod

import lie_nn as lie


@multimethod
def change_algebra(rep: lie.Rep, Q: np.ndarray) -> lie.GenericRep:
    """Apply change of basis to algebra.

    .. math::

        X'_i = Q_{ia} X_a

    .. math::

        A'_{ijk} = Q_{ia} Q_{jb} A_{abc} Q^{-1}_{ck}

    """
    iQ = np.linalg.pinv(Q)
    return lie.GenericRep(
        A=np.einsum("ia,jb,abc,ck->ijk", Q, Q, rep.A, iQ),
        X=np.einsum("ia,auv->iuv", Q, rep.X),
        H=rep.H,
    )


@multimethod
def change_algebra(rep: lie.QRep, Q: np.ndarray) -> lie.Rep:  # noqa: F811
    return lie.change_basis(rep.Q, change_algebra(rep.rep, Q))


@multimethod
def change_algebra(rep: lie.SumRep, Q: np.ndarray) -> lie.Rep:  # noqa: F811
    return lie.direct_sum(*[change_algebra(subrep, Q) for subrep in rep.reps])


@multimethod
def change_algebra(rep: lie.MulRep, Q: np.ndarray) -> lie.Rep:  # noqa: F811
    return lie.multiply(rep.mul, change_algebra(rep.rep, Q))
