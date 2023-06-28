import numpy as np
from multimethod import multimethod

import lie_nn as lie


@multimethod
def change_basis(Q: np.ndarray, rep: lie.Rep) -> lie.QRep:
    """Apply change of basis to generators.

    .. math::

        X' = Q X Q^{-1}

    .. math::

        v' = Q v

    """
    assert Q.shape == (rep.dim, rep.dim), (Q.shape, rep.dim)

    if np.allclose(Q.imag, 0.0, atol=1e-10):
        Q = Q.real

    if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
        return rep

    return lie.QRep(Q, rep, force=True)


@multimethod
def change_basis(Q: np.ndarray, rep: lie.QRep) -> lie.Rep:  # noqa: F811
    return change_basis(Q @ rep.Q, rep.rep)
