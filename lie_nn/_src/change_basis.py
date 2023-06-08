import numpy as np
from multipledispatch import dispatch

from .rep import PRep, QRep, Rep, SumRep

# change_basis:


@dispatch(object, Rep)
def change_basis(Q: np.ndarray, rep: Rep) -> Rep:
    """Apply change of basis to generators.

    .. math::

        X' = Q X Q^{-1}

    .. math::

        v' = Q v

    """
    assert Q.shape == (rep.dim, rep.dim)

    if np.allclose(Q.imag, 0.0, atol=1e-10):
        Q = Q.real

    if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
        return rep

    return QRep(Q, rep, force=True)


@dispatch(object, QRep)
def change_basis(Q: np.ndarray, rep: QRep) -> QRep:  # noqa: F811
    return change_basis(Q @ rep.Q, rep.rep)


# project:


@dispatch(object, Rep)
def project(U: np.ndarray, rep: Rep) -> Rep:
    """Project onto subspace.

    .. math::

        X' = U X U^H

    .. math::

        v' = U v

    """
    assert U.shape[1] == rep.dim

    if U.shape == (rep.dim, rep.dim):
        return change_basis(U, rep)

    if np.allclose(U.imag, 0.0, atol=1e-10):
        U = U.real

    return PRep(U, rep, force=True)


@dispatch(object, QRep)
def project(U: np.ndarray, qrep: QRep) -> Rep:  # noqa: F811
    if U.shape == (qrep.dim, qrep.dim):
        return change_basis(U, qrep)

    u, s, vh = np.linalg.svd(U @ qrep.Q, full_matrices=False)
    Q = np.einsum("ij,j->ij", u, s)
    U = vh
    return change_basis(Q, project(U, qrep.rep))


@dispatch(object, SumRep)
def project(U: np.ndarray, sumrep: SumRep) -> Rep:  # noqa: F811
    raise NotImplementedError
    # TODO
    # proj(rep1 + rep2) = proj(rep1) + proj(rep2)
