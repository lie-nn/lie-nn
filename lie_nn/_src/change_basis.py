import numpy as np
from multipledispatch import dispatch

from .rep import QRep, Rep


@dispatch(object, Rep)
def change_basis(Q: np.ndarray, rep: Rep) -> Rep:
    """Apply change of basis to generators.

    .. math::

        X' = Q X Q^{-1}

    .. math::

        v' = Q v

    """
    if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
        return rep

    return QRep(Q, rep)


@dispatch(object, QRep)
def change_basis(Q: np.ndarray, rep: QRep) -> QRep:  # noqa: F811
    Q = Q @ rep.Q

    if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
        return rep.rep

    return QRep(Q, rep.rep)
