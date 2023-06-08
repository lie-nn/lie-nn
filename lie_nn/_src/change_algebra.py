import numpy as np
from multipledispatch import dispatch

from .change_basis import change_basis
from .direct_sum import direct_sum
from .multiply import multiply
from .rep import GenericRep, MulRep, QRep, Rep, SumRep


@dispatch(Rep, object)
def change_algebra(rep: Rep, Q: np.ndarray) -> GenericRep:
    """Apply change of basis to algebra.

    .. math::

        X'_i = Q_{ia} X_a

    .. math::

        A'_{ijk} = Q_{ia} Q_{jb} A_{abc} Q^{-1}_{ck}

    """
    iQ = np.linalg.pinv(Q)
    return GenericRep(
        A=np.einsum("ia,jb,abc,ck->ijk", Q, Q, rep.A, iQ),
        X=np.einsum("ia,auv->iuv", Q, rep.X),
        H=rep.H,
    )


@dispatch(QRep, object)
def change_algebra(rep: QRep, Q: np.ndarray) -> Rep:  # noqa: F811
    return change_basis(rep.Q, change_algebra(rep.rep, Q))


@dispatch(SumRep, object)
def change_algebra(rep: SumRep, Q: np.ndarray) -> Rep:  # noqa: F811
    return direct_sum(*[change_algebra(subrep, Q) for subrep in rep.reps])


@dispatch(MulRep, object)
def change_algebra(rep: MulRep, Q: np.ndarray) -> Rep:  # noqa: F811
    return multiply(rep.mul, change_algebra(rep.rep, Q))
