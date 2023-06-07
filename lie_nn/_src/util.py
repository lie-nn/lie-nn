from functools import reduce
from typing import List, Optional, Tuple, Union

import numpy as np
import sympy as sp


def prod(list_of_numbers: List[Union[int, float]]) -> Union[int, float]:
    """Product of a list of numbers."""
    return reduce(lambda x, y: x * y, list_of_numbers, 1)


def is_integer(x: float) -> bool:
    return x == round(x)


def is_half_integer(x: float) -> bool:
    return 2 * x == round(2 * x)


def normalize_integer_ratio(n, d):
    g = np.gcd(n, d)
    g = np.where(d < 0, -g, g)
    return n // g, d // g


def _as_approx_integer_ratio(x):
    # only for 0 < x <= 1
    assert np.all(0 < x), x
    assert np.all(x <= 1), x
    big = 1 << 52 - 1  # mantissa is 52 bits

    n = np.floor(x * big).astype(np.int64)
    d = np.round(n / x).astype(np.int64)
    d = np.where(d == 0, 1, d)  # the case when x is tiny but not zero
    return n, d


def as_approx_integer_ratio(x):
    x = np.asarray(x)
    assert x.dtype == np.float64
    sign = np.sign(x).astype(np.int64)
    x = np.abs(x)
    x_ = np.where(x == 0.0, 1.0, x)

    n, d = np.where(
        x <= 1,
        _as_approx_integer_ratio(np.where(x_ <= 1.0, x_, 1.0)),
        _as_approx_integer_ratio(np.where(1 / x_ <= 1.0, 1 / x_, 1.0))[::-1],
    )
    n = np.where(x == 0.0, 0, n)
    d = np.where(x == 0.0, 1, d)
    assert np.all(d > 0), d
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
    with np.errstate(over="ignore"):
        mask = np.abs(d1 * (n2 * d0 - n0 * d2)) <= np.abs(d2 * (n1 * d0 - n0 * d1))
    return np.where(
        d0 < max_denominator,
        (n0, d0),
        np.where(mask, (n2, d2), (n1, d1)),
    )


def _round_to_sqrt_rational(x, max_denominator):
    sign = np.sign(x)
    n, d = as_approx_integer_ratio(x**2)
    n, d = limit_denominator(n, d, max_denominator**2 + 1)
    return sign * np.sqrt(n / d)


def _round_to_sqrt_rational_sympy(x, max_denominator):
    sign = np.sign(x)
    n, d = as_approx_integer_ratio(x**2)
    n, d = limit_denominator(n, d, max_denominator**2 + 1)
    n, d = n.item(), d.item()
    x = sp.sqrt(sp.Number(n) / sp.Number(d))
    if sign < 0:
        x = -x
    return x


def round_to_sqrt_rational_sympy(x, max_denominator):
    x, y = np.real(x), np.imag(x)

    return _round_to_sqrt_rational_sympy(x, max_denominator) + 1j * _round_to_sqrt_rational_sympy(
        y, max_denominator
    )


def round_to_sqrt_rational(x: np.ndarray, max_denominator=4096) -> np.ndarray:
    x = np.array(x)
    if np.iscomplex(x).any():
        return _round_to_sqrt_rational(np.real(x), max_denominator) + 1j * _round_to_sqrt_rational(
            np.imag(x), max_denominator
        )
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
            out = fun(
                *[
                    arg if in_axis is None else np.take(arg, i, in_axis)
                    for arg, in_axis in zip(args, in_axes_)
                ]
            )
            output.append(out)

        if len(output) == 0:
            return np.empty(out_shape)

        if isinstance(output[0], tuple):
            return tuple(np.stack(list, axis=out_axes) for list in zip(*output))
        return np.stack(output, axis=out_axes)

    return f


def block_diagonal(As: List[np.array]):
    size1 = sum([As[i].shape[1] for i in range(len(As))])
    size2 = sum([As[i].shape[2] for i in range(len(As))])
    R = np.zeros((As[0].shape[0], size1, size2))
    shape_x = 0
    shape_y = 0
    for i in range(len(As)):
        R[:, shape_x : shape_x + As[i].shape[1], shape_y : shape_y + As[i].shape[2]] = As[i]
        shape_x += As[i].shape[1]
        shape_y += As[i].shape[2]
    return R


def commutator(A, B):
    return A @ B - B @ A


def kron(A, *BCD):
    if len(BCD) == 0:
        return A
    return np.kron(A, kron(*BCD))


def direct_sum(A, *BCD):
    r"""Direct sum of matrices.

    Args:
        A (np.ndarray): Matrix of shape (..., m, n).
        B (np.ndarray): Matrix of shape (..., p, q).

    Returns:
        np.ndarray: Matrix of shape (..., m + p, n + q).
    """
    if len(BCD) == 0:
        return A
    B = direct_sum(*BCD)

    shape = np.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    A = np.broadcast_to(A, shape + A.shape[-2:])
    B = np.broadcast_to(B, shape + B.shape[-2:])

    m, n = A.shape[-2:]
    p, q = B.shape[-2:]

    output = np.zeros(shape=shape + (m + p, n + q), dtype=np.complex128)
    output[..., :m, :n] = A
    output[..., m:, n:] = B
    return output


def gram_schmidt(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
    """
    Orthogonalize a matrix using the Gram-Schmidt process.
    """
    assert A.ndim == 2, "Gram-Schmidt process only works for matrices."
    assert A.dtype in [
        np.float64,
        np.complex128,
    ], "Gram-Schmidt process only works for float64 matrices."
    Q = []
    for i in range(A.shape[0]):
        v = np.copy(A[i])
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            v = round_fn(v / norm)
            Q += [v]
    return np.stack(Q) if len(Q) > 0 else np.empty((0, A.shape[1]))


def extend_basis(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x, returns="Q") -> np.ndarray:
    """Add rows to A to make it full rank.

    Args:
        A (np.ndarray): Matrix of shape (m, n) with m <= n.
        epsilon (float): Tolerance for rank detection.
        round_fn (Callable): Function to round numbers.
        returns (str): What to return. Can be "Q" or "E".
            "Q" returns the complete orthogonal basis.
            "E" returns the matrix that extends A to a full rank matrix.

    Returns:
        np.ndarray: Matrix of shape (n, n) (if returns=Q) or (n - m, n) (if returns=E).
    """
    Q = gram_schmidt(A, epsilon=epsilon, round_fn=round_fn)
    Q = [q for q in Q]
    E = []
    for v in np.eye(A.shape[1], dtype=A.dtype):
        for w in Q:
            v -= np.dot(np.conj(w), v) * w
        norm = np.linalg.norm(v)
        if norm > epsilon:
            v = round_fn(v / norm)
            Q += [v]
            E += [v]
    if returns == "Q":
        return np.stack(Q)
    if returns == "E":
        return np.stack(E)


def nullspace(A: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x) -> np.ndarray:
    r"""Compute the null space of a matrix.

    .. math::

        \mathbf{A} \mathbf{X}^T = 0

    Args:
        A: Matrix to compute null space of.
        epsilon: The tolerance for the eigenvalue.
        round_fn: Function to round the numbers to.

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
    X = gram_schmidt(X, round_fn=round_fn, epsilon=epsilon)
    return X


def sequential_nullspace(
    gen_A: List[np.ndarray], dim_null_space: int, *, epsilon=1e-4, round_fn=lambda x: x
) -> np.ndarray:
    r"""Compute the null space of a list of matrices.

    .. math::

        \mathbf{A}_1 \mathbf{X}^T = 0
        \mathbf{A}_2 \mathbf{X}^T = 0

    Args:
        gen_A: List of matrices to compute null space of. Can be a generator.
        dim_null_space: The dimension of the null space. The algorithm will stop when
                        the null space has this dimension.
        epsilon: The tolerance for the eigenvalue.
        round_fn: Function to round the numbers to.

    Returns:
        The null space
    """
    S = None
    n = 0
    m = 0
    for A in gen_A:
        if S is None:
            S = nullspace(A, epsilon=epsilon, round_fn=round_fn)  # (num_null_space, dim_total)
        else:
            S = (
                nullspace(A @ S.T, epsilon=epsilon, round_fn=round_fn) @ S
            )  # (num_null_space, dim_total)

        n += 1
        m += A.shape[0]

        if S.shape[0] <= dim_null_space:
            return S
    raise ValueError(
        (
            f"Could not compute null space of dimension {dim_null_space}. "
            "Not enough constraints available. "
            f"{n} elements in the generator, {m} dimensions constrained."
        )
    )


def infer_change_of_basis(
    X1: np.ndarray, X2: np.ndarray, *, epsilon=1e-4, round_fn=lambda x: x
) -> np.ndarray:
    r"""
    Compute the change of basis matrix from X1 to X2.

    .. math::
        \mathbf{X_1} \mathbf{S} = \mathbf{S} \mathbf{X_2}

    Args:
        X1: Ensemble of matrices of shape (n, d1, d1).
        X2: Ensemble of matrices of shape (n, d2, d2).

    Returns:
        The change of basis S of shape (n_solutions, d1, d2).
    """
    assert X1.dtype in [
        np.float64,
        np.complex128,
    ], "Change of basis only works for float64 matrices."
    assert X2.dtype in [
        np.float64,
        np.complex128,
    ], "Change of basis only works for float64 matrices."

    if X1.ndim == 2:
        X1 = X1[np.newaxis]
    if X2.ndim == 2:
        X2 = X2[np.newaxis]

    n, d1, _ = X1.shape
    _, d2, _ = X2.shape
    assert X1.shape == (n, d1, d1)
    assert X2.shape == (n, d2, d2)

    As = []
    for x1, x2 in zip(X1, X2):
        # X1 @ S - S @ X2 = 0
        # X1 @ S - (X2.T @ S.T).T = 0
        # A @ vec(S) = 0
        As.append(np.kron(x1, np.eye(d2)) - np.kron(np.eye(d1), x2.T))

    A = np.concatenate(As, axis=0)  # (n * d1 * d2, d1 * d2)
    S = nullspace(A, epsilon=epsilon, round_fn=round_fn)  # (m, d1 * d2)
    S = S.reshape(-1, d1, d2)

    # assert np.allclose(X1 @ S[0], S[0] @ X2)
    return S


def basis_intersection(
    basis1: np.ndarray, basis2: np.ndarray, *, epsilon=1e-5, round_fn=lambda x: x
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the intersection of two bases

    Args:
        basis1 (np.ndarray): A basis, shape ``(n1, ...)``.
        basis2 (np.ndarray): Another basis, shape ``(n2, ...)``.
        epsilon (float, optional): Tolerance for the norm of the vectors. Defaults to 1e-4.
        round_fn (function, optional): Function to round the vectors. Defaults to lambda x: x.

    Returns:
        np.ndarray: A projection matrix that projects vectors of the first basis in
                    the intersection of the two bases.
                    Shape ``(dim_intersection, n1)``
        np.ndarray: A projection matrix that projects vectors of the second basis in
                    the intersection of the two bases.
                    Shape ``(dim_intersection, n2)``

    Example:
        >>> basis1 = np.array([[1, 0, 0], [0, 0, 1.0]])
        >>> basis2 = np.array([[1, 1, 0], [0, 1, 0.0]])
        >>> P1, P2 = basis_intersection(basis1, basis2)
        >>> P1 @ basis1
        array([[1., 0., 0.]])
    """
    assert basis1.shape[1:] == basis2.shape[1:]
    basis1 = np.reshape(basis1, (basis1.shape[0], -1))
    basis2 = np.reshape(basis2, (basis2.shape[0], -1))

    p = np.concatenate(
        [
            np.concatenate([basis1 @ basis1.T, -basis1 @ basis2.T], axis=1),
            np.concatenate([-basis2 @ basis1.T, basis2 @ basis2.T], axis=1),
        ],
        axis=0,
    )
    p = round_fn(p)

    w, v = np.linalg.eigh(p)
    v = v[:, w < epsilon]

    x1 = v[: basis1.shape[0], :]
    x1 = gram_schmidt(x1 @ x1.T, epsilon=epsilon, round_fn=round_fn)

    x2 = v[basis1.shape[0] :, :]
    x2 = gram_schmidt(x2 @ x2.T, epsilon=epsilon, round_fn=round_fn)
    return x1, x2


def check_algebra_vs_generators(
    A: np.ndarray, X: np.ndarray, rtol: float = 1e-10, atol: float = 1e-10, assert_: bool = False
):
    left_side = vmap(vmap(commutator, (0, None), 0), (None, 0), 1)(X, X)
    right_side = np.einsum("ijk,kab->ijab", A, X)
    if assert_:
        np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)
    return np.allclose(left_side, right_side, rtol=rtol, atol=atol)


def infer_algebra_from_generators(
    X: np.ndarray, *, round_fn=lambda x: x, rtol: float = 1e-10, atol: float = 1e-10
) -> Optional[np.ndarray]:
    """Infer the algebra from the generators.

    .. math::

        [X_i, X_j] = A_ijk X_k

    Args:
        X (np.ndarray): The generators of the algebra. Shape ``(n, d, d)``.
        round_fn (function, optional): Function to round the matrices. Defaults to the identity.
        rtol (float, optional): Relative tolerance to test the validity of the algebra.
                                Defaults to 1e-10.
        atol (float, optional): Absolute tolerance to test the validity of the algebra.
                                Defaults to 1e-10.

    Returns:
        np.ndarray: If successful, the algebra. Shape ``(n, n, n)``.
    """
    n = X.shape[0]
    pinv = np.linalg.pinv(X.reshape((n, -1)))

    algebra = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            xij = commutator(X[i], X[j])
            algebra[i, j] = xij.reshape(-1) @ pinv

    algebra = round_fn(algebra)

    if check_algebra_vs_generators(algebra, X, rtol=rtol, atol=atol):
        return algebra
    else:
        return None


def permutation_sign(p: Tuple[int, ...]) -> int:
    if len(p) == 1:
        return 1

    trans = 0
    for i in range(0, len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                trans += 1

    return 1 if (trans % 2) == 0 else -1


def unique_with_tol(a: np.array, *, tol: float):
    """Find unique elements of an array with a tolerance.
    Input:
        a: np.array of shape num_elements x d1 x ... x dm of which to find the unique elements
        tol: tolerance
    Output:
        centers: np.array of shape num_clusters x d1 x ... x dm containing
                 the centers of the clusters
        inverses: np.array of shape num_elements containing the index of the corresponding center
                  for each element of a
    Raises:
        ValueError: if the cluster are not clearly distinct
    Note:
        this function is "stable", the first element always belongs to the
        first cluster, the second element not in the first cluster belongs to the
        second cluster, etc.
    """
    assert a.ndim >= 1
    shape = a.shape
    a = a.reshape(len(a), -1)

    distances = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    inverses = -1 * np.ones(len(a), dtype=int)
    index = 0

    while True:
        (m,) = np.nonzero(inverses == -1)
        if len(m) == 0:
            break
        i = m[0]

        if np.any(inverses[distances[i] < tol] != -1):
            raise ValueError("The clusters are not clearly distinct.")

        inverses[distances[i] < tol] = index

        index += 1

    centers = np.zeros((np.max(inverses) + 1, a.shape[1]), dtype=a.dtype)
    np.add.at(centers, inverses, a)
    centers /= np.bincount(inverses)[:, None]

    centers = centers.reshape(len(centers), *shape[1:])
    return centers, inverses


def eigenspaces(
    val: np.ndarray, vec: np.ndarray, *, epsilon: float = 1e-6
) -> List[Tuple[float, np.ndarray]]:
    """Regroup eigenvectors by eigenvalues.
    Input:
        val: eigenvalues (output of np.linalg.eig)
        vec: eigenvectors (output of np.linalg.eig)
        tol: tolerance for the eigenvalues similarity
    Output:
        list of (eigenvalue, eigenvectors) tuples
    """
    unique_val, i = unique_with_tol(val, tol=epsilon)
    return [(val, vec[:, i == j]) for j, val in enumerate(unique_val)]


def decompose_rep_into_irreps(
    X: np.array, *, epsilon: float = 1e-10, round_fn=lambda x: x
) -> List[np.array]:
    """Decomposes representation into irreducible representations.
    Input:
        X: np.array [num_gen, d, d] - generators of a representation.
    Output:
        Ys: List[np.array] - list of generators of irreducible representations.
    """
    Q = infer_change_of_basis(X, X, epsilon=epsilon, round_fn=round_fn)  # X @ Q == Q @ X
    w = np.random.rand(len(Q))
    M = np.einsum("n,nij->ij", w, Q)
    val, vec = np.linalg.eig(M)
    stable_spaces = eigenspaces(val, vec, epsilon=epsilon)

    Ys = []
    for _, W in stable_spaces:
        B = gram_schmidt(W.T, epsilon=epsilon)  # Make an orthonormal projector
        B = gram_schmidt(B.T.conj() @ B, epsilon=epsilon, round_fn=round_fn)  # Make it sparse!!

        Ys += [B @ X @ B.T.conj()]
    return Ys


def is_irreducible(X: np.array, *, epsilon: float = 1e-10) -> bool:
    Q = infer_change_of_basis(X, X, epsilon=epsilon)  # X @ Q == Q @ X
    return len(Q) == 1


def regular_representation(table: np.array) -> np.array:
    """Returns regular representation for group represented by a multiplication table.
    Input:
        table: np.array [n, n] where table[i, j] = k means i * j = k.
    Output:
        Regular representation. array [n, n, n] where reg_rep[i, :, :] = D(i) and D(i)e_j = e_{ij}.
                                Equivalently, D(g) |h> = |gh>
    """
    n, _ = table.shape
    g, h = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    gh = table
    # D[g] |h> = |gh>     =>    <gh| D[g] |h> = 1
    reg_rep = np.zeros((n, n, n))
    reg_rep[g, gh, h] = 1
    return reg_rep
