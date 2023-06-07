import numpy as np

from .rep import GenericRep, Rep


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
