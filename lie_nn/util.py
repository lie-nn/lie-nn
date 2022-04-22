import fractions
from functools import partial

import numpy as np


def is_integer(x: float) -> bool:
    return x == round(x)


def is_half_integer(x: float) -> bool:
    return 2 * x == round(2 * x)


@partial(np.vectorize, otypes=[np.float64])
def round_to_sqrt_rational(x: float) -> float:
    sign = 1 if x >= 0 else -1
    return sign * fractions.Fraction(x ** 2).limit_denominator() ** 0.5


def vmap(
    fun,
    in_axes=0,
    out_axes=0,
):
    def f(*args):
        in_axes_ = in_axes
        if isinstance(in_axes_, int):
            in_axes_ = (in_axes_,) * len(args)
        assert len(in_axes_) == len(args)

        dims = set()
        for arg, in_axis in zip(args, in_axes_):
            if in_axis is not None:
                dims.add(arg.shape[in_axis])
        assert len(dims) == 1
        dim = dims.pop()

        output = []
        for i in range(dim):
            out = fun(*[arg if in_axis is None else np.take(arg, i, in_axis) for arg, in_axis in zip(args, in_axes_)])
            output.append(out)

        if isinstance(output[0], tuple):
            return tuple(np.stack(list, axis=out_axes) for list in zip(*output))
        return np.stack(output, axis=out_axes)

    return f


def commutator(A, B):
    return A @ B - B @ A


def kron(A, *BCD):
    if len(BCD) == 0:
        return A
    return np.kron(A, kron(*BCD))


def gram_schmidt(A: np.ndarray, epsilon=1e-4) -> np.ndarray:
    """
    Orthogonalize a matrix using the Gram-Schmidt process.
    """
    assert A.ndim == 2, "Gram-Schmidt process only works for matrices."
    assert A.dtype in [np.float64, np.complex128], "Gram-Schmidt process only works for float64 matrices."
    Q = []
    for i in range(A.shape[0]):
        v = A[i]
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            Q += [v / norm]
    return np.stack(Q) if len(Q) > 0 else np.empty((0, A.shape[1]))


def null_space(A: np.ndarray, epsilon=1e-4) -> np.ndarray:
    r"""
    Compute the null space of a matrix.

    .. math::
        \mathbf{A} \mathbf{X}^T = 0

    Args:
        A: Matrix to compute null space of.
        epsilon: The tolerance for the eigenvalue.

    Returns:
        The null space of A.
    """
    assert A.ndim == 2, "Null space only works for matrices."
    assert A.dtype in [np.float64, np.complex128], "Null space only works for float64 matrices."

    # Q, R = np.linalg.qr(A.T)
    # # assert np.allclose(R.T @ Q.T, S)
    # X = Q.T[np.abs(np.diag(R)) < epsilon]
    # X = np.conj(X)

    A = np.conj(A.T) @ A
    val, vec = np.linalg.eigh(A)
    X = vec.T[np.abs(val) < epsilon]

    X = gram_schmidt(X.T @ X)
    return X


def change_of_basis(X1: np.ndarray, X2: np.ndarray, epsilon=1e-4) -> np.ndarray:
    r"""
    Compute the change of basis matrix from X1 to X2.

    .. math::
        \mathbf{X_1} \mathbf{S} = \mathbf{S} \mathbf{X_2}

    Args:
        X1: Ensemble of matrices.
        X2: Ensemble of matrices.

    Returns:
        The change of basis S.
    """
    assert X1.dtype in [np.float64, np.complex128], "Change of basis only works for float64 matrices."
    assert X2.dtype in [np.float64, np.complex128], "Change of basis only works for float64 matrices."

    n, d1, _ = X1.shape
    _, d2, _ = X2.shape
    assert X1.shape == (n, d1, d1)
    assert X2.shape == (n, d2, d2)

    A = vmap(lambda x1, x2: kron(np.eye(d2), x1) - kron(x2.T, np.eye(d1)))(X1, X2)
    A = A.reshape(n * d2 * d1, d2 * d1)
    S = null_space(A, epsilon)
    S = S.reshape(-1, d2, d1)
    S = np.swapaxes(S, 1, 2)

    # assert np.allclose(X1 @ S[0], S[0] @ X2)
    return S
