import numpy as np


def is_integer(x: float) -> bool:
    return x == round(x)


def is_half_integer(x: float) -> bool:
    return 2 * x == round(2 * x)


def normalize_integer_ratio(n, d):
    g = np.gcd(n, d)
    g = np.where(d < 0, -g, g)
    return n // g, d // g


def _as_approx_integer_ratio(x):
    # only for 0 <= x <= 1
    big = 1 << 52 - 1  # mantissa is 52 bits

    n = np.floor(x * big).astype(np.int64)
    with np.errstate(invalid="ignore"):
        d = np.round(n / x).astype(np.int64)
    d = np.where(n == 0, np.ones(d.shape, dtype=np.int64), d)
    return n, d


def as_approx_integer_ratio(x):
    assert x.dtype == np.float64
    sign = np.sign(x).astype(np.int64)
    x = np.abs(x)

    with np.errstate(divide="ignore", over="ignore"):
        n, d = np.where(
            x <= 1,
            _as_approx_integer_ratio(x),
            _as_approx_integer_ratio(1 / x)[::-1],
        )
    return normalize_integer_ratio(sign * n, d)


def limit_denominator(n, d, max_denominator=1_000_000):
    # (n, d) = must be normalized
    n0, d0 = n, d
    p0, q0, p1, q1 = np.zeros_like(n), np.ones_like(n), np.ones_like(n), np.zeros_like(n)
    while True:
        a = n // d
        q2 = q0 + a * q1
        stop = (q2 > max_denominator) | (d0 <= max_denominator)
        if np.all(stop):
            break
        p0, q0, p1, q1 = np.where(stop, (p0, q0, p1, q1), (p1, q1, p0 + a * p1, q2))
        n, d = np.where(stop, (n, d), (d, n - a * d))

    with np.errstate(divide="ignore"):
        k = (max_denominator - q0) // q1
    n1, d1 = p0 + k * p1, q0 + k * q1
    n2, d2 = p1, q1
    return np.where(
        d0 < max_denominator,
        (n0, d0),
        np.where(np.abs(d1 * (n2 * d0 - n0 * d2)) <= np.abs(d2 * (n1 * d0 - n0 * d1)), (n2, d2), (n1, d1)),
    )


def _round_to_sqrt_rational(x, max_denominator):
    sign = np.sign(x)
    n, d = as_approx_integer_ratio(x**2)
    n, d = limit_denominator(n, d, max_denominator**2 + 1)
    return sign * np.sqrt(n / d)


def round_to_sqrt_rational(x: np.ndarray, max_denominator=4096) -> np.ndarray:
    x = np.array(x)
    if np.iscomplex(x).any():
        return _round_to_sqrt_rational(np.real(x), max_denominator) + 1j * _round_to_sqrt_rational(np.imag(x), max_denominator)
    return _round_to_sqrt_rational(np.real(x), max_denominator)


def vmap(
    fun,
    in_axes=0,
    out_axes=0,
    *,
    out_shape=None,
):
    if out_shape is not None:
        out_shape = list(out_shape)
        out_shape.insert(out_axes, 0)

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

        if len(output) == 0:
            return np.empty(out_shape)

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


def direct_sum(A, *BCD):
    if len(BCD) == 0:
        return A
    B = direct_sum(*BCD)
    a = A.shape[0]
    b = B.shape[0]
    output = np.zeros_like(A, shape=(a + b, a + b))
    output[:a, :a] = A
    output[a:, a:] = B
    return output


def gram_schmidt(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
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
            v = round_fn(v / norm)
            Q += [v]
    return np.stack(Q) if len(Q) > 0 else np.empty((0, A.shape[1]))


def extend_basis(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
    """
    Add rows to A to make it full rank.
    """
    Q = gram_schmidt(A, epsilon=epsilon, round_fn=round_fn)
    Q = [q for q in Q]
    for v in np.eye(A.shape[1], dtype=A.dtype):
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            v = round_fn(v / norm)
            Q += [v]
    return np.stack(Q)


def null_space(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
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
    A = round_fn(A)
    val, vec = np.linalg.eigh(A)
    X = vec.T[np.abs(val) < epsilon]
    X = np.conj(X.T) @ X
    X = round_fn(X)
    X = gram_schmidt(X, round_fn=round_fn)
    return X


def change_of_basis(X1: np.ndarray, X2: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
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

    if X1.ndim == 2:
        X1 = X1[np.newaxis]
    if X2.ndim == 2:
        X2 = X2[np.newaxis]

    n, d1, _ = X1.shape
    _, d2, _ = X2.shape
    assert X1.shape == (n, d1, d1)
    assert X2.shape == (n, d2, d2)

    A = vmap(lambda x1, x2: kron(np.eye(d2), x1) - kron(x2.T, np.eye(d1)))(X1, X2)
    A = A.reshape(n * d2 * d1, d2 * d1)
    S = null_space(A, epsilon=epsilon, round_fn=round_fn)
    S = S.reshape(-1, d2, d1)
    S = np.swapaxes(S, 1, 2)

    # assert np.allclose(X1 @ S[0], S[0] @ X2)
    return S
