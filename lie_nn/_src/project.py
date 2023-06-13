import numpy as np
from multimethod import multimethod

from .change_basis import change_basis
from .rep import PRep, QRep, Rep  # , SumRep


@multimethod
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


@multimethod
def project(U: np.ndarray, qrep: QRep) -> Rep:  # noqa: F811
    assert U.shape[1] == qrep.dim

    if U.shape == (qrep.dim, qrep.dim):
        return change_basis(U, qrep)

    if U.shape[0] < U.shape[1]:
        u, s, vh = np.linalg.svd(U @ qrep.Q, full_matrices=False)
        Q = np.einsum("ij,j->ij", u, s)
        U = vh
        return change_basis(Q, project(U, qrep.rep))

    if U.shape[0] > U.shape[1]:
        return project(U @ qrep.Q, qrep.rep)

    raise NotImplementedError


# @multimethod
# def project(U: np.ndarray, sumrep: SumRep) -> Rep:  # noqa: F811
#     assert U.shape[1] == sumrep.dim

#     base = np.linalg.pinv(U).T

#     for r in sumrep.reps:  # independent irreps
#         R = np.eye(rep.dim)[i:j]
#         P, _ = basis_intersection(base, R)


#     # TODO: can we do this better? without GenericRep?
#     newreps = []
#     i = 0
#     for subrep in sumrep.reps:
#         j = i + subrep.dim
#         subU = U[:, i:j]
#         subrep = project(subU, subrep)
#         newreps.append(subrep)
#         i = j

#     return GenericRep(
#         sumrep.A,
#         sum(rep.X for rep in newreps),
#         sum(rep.H for rep in newreps),
#     )
