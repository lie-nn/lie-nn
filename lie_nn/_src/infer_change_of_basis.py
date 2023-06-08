import numpy as np
from multimethod import multimethod

from .rep import QRep, Rep
from .util import infer_change_of_basis as _infer_change_of_basis


@multimethod
def infer_change_of_basis(rep1: Rep, rep2: Rep, *, round_fn=lambda x: x) -> np.ndarray:
    r"""Infers the change of basis matrix between two representations.

    .. math::

        v_2 = Q v_1

    .. math::

        Q \rho_1 = \rho_2 Q

    Args:
        rep1: A representation.
        rep2: A representation.
        round_fn (optional): A rounding function used in numerical computations.

    Returns:
        The change of basis matrix ``Q``.
    """
    # Check the group structure
    assert np.allclose(rep1.A, rep2.A)

    Y1 = np.concatenate([rep1.X, rep1.H])
    Y2 = np.concatenate([rep2.X, rep2.H])

    A = _infer_change_of_basis(Y2, Y1, round_fn=round_fn)
    np.testing.assert_allclose(
        np.einsum("aij,bjk->abik", Y2, A),
        np.einsum("bij,ajk->abik", A, Y1),
        rtol=1e-8,
        atol=1e-8,
    )

    assert A.dtype in [np.float64, np.complex128]

    A = A * np.sqrt(rep2.dim)
    A = round_fn(A)
    return A


@multimethod
def infer_change_of_basis(  # noqa: F811
    rep1: QRep, rep2: Rep, *, round_fn=lambda x: x
) -> np.ndarray:
    r"""
    Q \rho_1 = \rho_2 Q
    (Q q) \rho_1 q^{-1} = \rho_2 Q
    """
    inv = np.linalg.pinv(rep1.Q)
    return infer_change_of_basis(rep1.rep, rep2, round_fn=round_fn) @ inv


@multimethod
def infer_change_of_basis(  # noqa: F811
    rep1: Rep, rep2: QRep, *, round_fn=lambda x: x
) -> np.ndarray:
    r"""
    Q \rho_1 = \rho_2 Q
    Q \rho_1 = q \rho_2 (q^{-1} Q)
    """
    return rep2.Q @ infer_change_of_basis(rep1, rep2.rep, round_fn=round_fn)


@multimethod
def infer_change_of_basis(  # noqa: F811
    rep1: QRep, rep2: QRep, *, round_fn=lambda x: x
) -> np.ndarray:
    r"""
    Q \rho_1 = \rho_2 Q
    Q q1 \rho_1 q1^{-1} = q2 \rho_2 q2^{-1} Q
    (q2^{-1} Q q1) \rho_1 = \rho_2 q2^{-1} Q q1
    """
    inv1 = np.linalg.pinv(rep1.Q)
    return rep2.Q @ infer_change_of_basis(rep1.rep, rep2.rep, round_fn=round_fn) @ inv1
