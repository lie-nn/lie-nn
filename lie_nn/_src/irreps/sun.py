import itertools
import re
from dataclasses import dataclass
from fractions import Fraction
from operator import add
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..irrep import TabulatedIrrep
from ..util import commutator, nullspace, round_to_sqrt_rational

WEIGHT = Tuple[int, ...]
GT_PATTERN = Tuple[WEIGHT, ...]


def _is_valid_S(S: WEIGHT):
    return len(S) > 0 and all(s1 >= s2 for s1, s2 in zip(S, S[1:]))


def _assert_valid_S(S: WEIGHT):
    """
    (3, 3, 2) is a valid S.
    (3, 2, 3) is not a valid S.
    """
    assert _is_valid_S(S)


def dim(S: WEIGHT) -> int:
    """Return the number of possible GT-patterns with first line S."""
    # A numerical algorithm for the explicit calculation of SU(N) and SL(N, C)
    # Clebsch-Gordan coefficients Arne Alex, Matthias Kalus, Alan Huckleberry
    # and Jan von Delft Eq 22.
    _assert_valid_S(S)
    d = 1
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d *= 1 + Fraction(S[i] - S[j], j - i)
    assert d.denominator == 1
    return int(d)


assert dim((3, 3, 1, 0)) == sum(dim((3, a, b)) for a in [3, 2, 1] for b in [1, 0])


def S_to_Ss(S: WEIGHT) -> Iterator[WEIGHT]:
    """Produce all possible next lines of S.

    >>> list(S_to_Ss((3, 3, 1, 0)))
    [(3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1), (3, 3, 0), (3, 3, 1)]
    """
    _assert_valid_S(S)
    if len(S) == 1:
        return
    diff = [s1 - s2 for s1, s2 in zip(S, S[1:])]
    for x in itertools.product(*[reversed(range(s + 1)) for s in diff]):
        yield tuple(S[i] - x[i] for i in range(len(S) - 1))


def S_to_Ms(S: WEIGHT) -> Iterator[GT_PATTERN]:
    """Produce all possible GT-patterns with first line S.

    >>> list(S_to_Ms((2, 0)))
    [((2, 0), (0,)), ((2, 0), (1,)), ((2, 0), (2,))]
    """
    _assert_valid_S(S)
    if len(S) == 1:
        yield (S,)
    for Snext in S_to_Ss(S):
        for M in S_to_Ms(Snext):
            yield (S,) + M


assert dim((3, 3, 1, 0)) == len(list(S_to_Ms((3, 3, 1, 0))))


def index_to_M(S: WEIGHT, index: int) -> GT_PATTERN:
    """Given an index, return the corresponding GT-pattern."""
    _assert_valid_S(S)
    assert index < dim(S)
    if len(S) == 1:
        return (S,)
    for Snext in S_to_Ss(S):
        d = dim(Snext)
        if index < d:
            return (S,) + index_to_M(Snext, index)
        index -= d


def M_to_index(M: GT_PATTERN) -> int:
    """Given a GT-pattern, return the corresponding index."""
    S = M[0]
    _assert_valid_S(S)
    if len(S) == 1:
        return 0
    index = 0
    for Snext in S_to_Ss(S):
        if M[1] == Snext:
            return index + M_to_index(M[1:])
        index += dim(Snext)


for i in range(dim((3, 2, 0))):
    assert M_to_index(index_to_M((3, 2, 0), i)) == i


def triangular_triplet(M: GT_PATTERN) -> Iterator[Tuple[int, int, int]]:
    """Produce all possible triplets of indices in M"""
    n = len(M)
    for i in range(1, n):
        for j in range(n - i):
            yield (M[i - 1][j], M[i - 1][j + 1], M[i][j])


def is_valid_M(M: GT_PATTERN):
    return all(m1 >= m3 >= m2 >= 0 for m1, m2, m3 in triangular_triplet(M))


def unique_pairs(n: int, start: int = 0) -> Iterator[Tuple[int, int]]:
    """Produce pairs of indexes in range(start,n)"""
    for i in range(start, n):
        for j in range(n - i):
            yield i, j


def M_to_sigma(M: GT_PATTERN) -> WEIGHT:
    return [sum(row) for row in M] + [0]


def M_to_p_weight(M: GT_PATTERN) -> WEIGHT:
    """The pattern weight of a GT-pattern."""
    sigma = M_to_sigma(M)[::-1]
    return tuple(s2 - s1 for s1, s2 in zip(sigma, sigma[1:]))[::-1]


def M_to_z_weight(M: GT_PATTERN) -> WEIGHT:
    """The pattern weight of a GT-pattern."""
    w = M_to_p_weight(M)[::-1]
    return tuple((s2 - s1) / 2 for s1, s2 in zip(w[1:], w))[::-1]


def mul_rep(S1: WEIGHT, S2: WEIGHT) -> Iterator[WEIGHT]:
    n = len(S1)
    for pattern in S_to_Ms(S1):
        t_weight = list(S2)
        for l, k in unique_pairs(n):
            try:
                t_weight[n - k - 1] += pattern[k][l] - pattern[k + 1][l]
            except IndexError:
                t_weight[n - k - 1] += pattern[k][l]
            if n - k - 1 >= 1 and t_weight[n - k - 1] > t_weight[n - k - 2]:
                t_weight = []
                break
        if t_weight:
            yield tuple(x - t_weight[-1] for x in t_weight)


def compute_coeff_upper(M: GT_PATTERN, k: int, l: int) -> float:
    """Compute the coefficient of the upper ladder operator."""
    n = len(M)
    num = 1
    for k_p in range(n - k + 1):
        num *= M[k - 1][k_p] - M[k][l] + l - k_p
    for k_p in range(n - k - 1):
        num *= M[k + 1][k_p] - M[k][l] + l - k_p - 1
    den = 1
    for k_p in range(n - k):
        if k_p == l:
            continue
        den *= (M[k][k_p] - M[k][l] + l - k_p) * (M[k][k_p] - M[k][l] + l - k_p - 1)
    assert -num / den >= 0
    return (-num / den) ** 0.5


def compute_coeff_lower(M: GT_PATTERN, k, l) -> float:
    """Compute the coefficient of the lower ladder."""
    n = len(M)
    num = 1
    for k_p in range(n - k + 1):
        num *= M[k - 1][k_p] - M[k][l] + l - k_p + 1
    for k_p in range(n - k - 1):
        num *= M[k + 1][k_p] - M[k][l] + l - k_p
    den = 1
    for k_p in range(n - k):
        if k_p == l:
            continue
        den *= (M[k][k_p] - M[k][l] + l - k_p + 1) * (M[k][k_p] - M[k][l] + l - k_p)
    assert -num / den >= 0
    return (-num / den) ** 0.5


def M_add_at_lk(M: GT_PATTERN, l: int, k: int, increment: int) -> Optional[GT_PATTERN]:
    """Add increment to the l-th row and k-th column of M."""
    M = tuple(
        tuple(M[i][j] + increment if (i, j) == (l, k) else M[i][j] for j in range(len(M[i])))
        for i in range(len(M))
    )
    return M if is_valid_M(M) else None


def upper_ladder(l: int, N: GT_PATTERN, M: GT_PATTERN) -> float:
    """<N| J(l)+ |M>

    Args:
        l: The index of the ladder operator. 0 <= l <= n-2.
        N: The left bracket GT-pattern.
        M: The right bracket GT-pattern.

    Returns:
        The coefficient of the ladder operator.
    """
    n = len(M)
    l = l + 1
    return sum(compute_coeff_upper(M, l, k) for k in range(n - l) if M_add_at_lk(M, l, k, 1) == N)


def lower_ladder(l: int, N: GT_PATTERN, M: GT_PATTERN) -> float:
    """<N| J(l)- |M>

    Args:
        l: The index of the ladder operator. 0 <= l <= n-2.
        N: The left bracket GT-pattern.
        M: The right bracket GT-pattern.

    Returns:
        The coefficient of the ladder operator.
    """
    n = len(M)
    l = l + 1
    return sum(compute_coeff_lower(M, l, k) for k in range(n - l) if M_add_at_lk(M, l, k, -1) == N)


def upper_ladder_matrices(S: WEIGHT) -> np.ndarray:
    return np.array(
        [
            [[upper_ladder(l, N, M) for M in S_to_Ms(S)] for N in S_to_Ms(S)]
            for l in range(len(S) - 1)
        ]
    ).reshape(len(S) - 1, dim(S), dim(S))


def lower_ladder_matrices(S: WEIGHT) -> np.ndarray:
    return np.array(
        [
            [[lower_ladder(l, N, M) for M in S_to_Ms(S)] for N in S_to_Ms(S)]
            for l in range(len(S) - 1)
        ]
    ).reshape(len(S) - 1, dim(S), dim(S))


def Jz_matrices(S: WEIGHT) -> np.ndarray:
    n = len(S)
    Jz = np.zeros((n - 1, dim(S), dim(S)))
    for i, M in enumerate(S_to_Ms(S)):
        z = M_to_z_weight(M)
        for l in range(n - 1):
            Jz[l, i, i] = z[l]
    return Jz


def construct_highest_weight_constraint(
    S1: WEIGHT, S2: WEIGHT, M_3_eldest: GT_PATTERN
) -> np.ndarray:
    n = len(M_3_eldest)  # SU(n)
    dimS1 = dim(S1)
    dimS2 = dim(S2)
    A = np.zeros((dimS1, dimS2, dimS1, dimS2, n - 1), dtype=np.float64)

    for m1 in range(dimS1):
        for m2 in range(dimS2):
            for n1 in range(dimS1):
                for n2 in range(dimS2):
                    for l in range(n - 1):
                        if n2 == m2:
                            A[m1, m2, n1, n2, l] += upper_ladder(
                                l, index_to_M(S1, n1), index_to_M(S1, m1)
                            )
                        if n1 == m1:
                            A[m1, m2, n1, n2, l] += upper_ladder(
                                l, index_to_M(S2, n2), index_to_M(S2, m2)
                            )

    A = A.reshape(dimS1, dimS2, -1)

    B = []

    for m1 in range(dimS1):
        for m2 in range(dimS2):
            W_1 = M_to_z_weight(index_to_M(S1, m1))
            W_2 = M_to_z_weight(index_to_M(S2, m2))
            W_eldest = M_to_z_weight(M_3_eldest)
            if tuple(map(add, W_1, W_2)) != W_eldest:
                b = np.zeros((dimS1, dimS2, 1))
                b[m1, m2] = 1
                B.append(b)

    return round_to_sqrt_rational(np.concatenate([A] + B, axis=2).reshape(dimS1 * dimS2, -1).T)


def clebsch_gordan_eldest(S1: WEIGHT, S2: WEIGHT, M_3_eldest: GT_PATTERN) -> np.ndarray:
    A = construct_highest_weight_constraint(S1, S2, M_3_eldest)
    return nullspace(A[:, ::-1], round_fn=round_to_sqrt_rational)[:, ::-1].reshape(
        -1, dim(S1), dim(S2)
    )


def search_state(M_list: List[GT_PATTERN]) -> Tuple[GT_PATTERN, GT_PATTERN, int]:
    M0 = M_list[0]
    n = len(M0)
    S3 = M0[0]

    for Mc in M_list:
        for Mp in S_to_Ms(S3):
            if Mp in M_list:
                continue

            for l in range(n - 1):
                if lower_ladder(l, Mp, Mc) != 0:
                    return (Mc, Mp, l)

    raise ValueError("No state found")


def construct_lower_cg(
    S1: WEIGHT,
    S2: WEIGHT,
    S3: WEIGHT,
    Mp: GT_PATTERN,
    Mc: GT_PATTERN,
    l: int,
    alpha: int,
    C: np.array,
) -> Tuple[np.ndarray, int]:
    dimS1 = dim(S1)
    dimS2 = dim(S2)
    mc, mp = M_to_index(Mc), M_to_index(Mp)
    CG = np.zeros((dimS1, dimS2))
    for m1 in range(dimS1):
        for m2 in range(dimS2):
            for n1 in range(dimS1):
                for n2 in range(dimS2):
                    if n2 == m2:
                        CG[n1, n2] += (
                            C[alpha, m1, m2, mc]
                            * lower_ladder(l, index_to_M(S1, n1), index_to_M(S1, m1))
                            * upper_ladder(l, index_to_M(S3, mc), index_to_M(S3, mp))
                        )
                    if n1 == m1:
                        CG[n1, n2] += (
                            C[alpha, m1, m2, mc]
                            * lower_ladder(l, index_to_M(S2, n2), index_to_M(S2, m2))
                            * upper_ladder(l, index_to_M(S3, mc), index_to_M(S3, mp))
                        )

    return CG / np.linalg.norm(CG)


def clebsch_gordan_matrix(S1: WEIGHT, S2: WEIGHT, S3: WEIGHT):
    dimS1 = dim(S1)
    dimS2 = dim(S2)
    dimS3 = dim(S3)
    M_eldest = tuple(S_to_Ms(S3))[-1]
    C_eldest = clebsch_gordan_eldest(S1, S2, M_eldest)
    C = np.concatenate(
        (
            np.zeros((C_eldest.shape[0], dimS1, dimS2, dimS3 - 1)),
            (C_eldest.reshape(C_eldest.shape[0], dimS1, dimS2, 1)),
        ),
        axis=-1,
    )
    M_list = [M_eldest]
    for alpha in range(C.shape[0]):
        while len(M_list) != dimS3:
            Mc, Mp, l = search_state(M_list)
            mp = M_to_index(Mp)
            C[alpha, :, :, mp] = construct_lower_cg(S1, S2, S3, Mp, Mc, l, alpha, C)
            M_list.append(Mp)
    return C


def E_basis(k: int, l: int, Jp, Jm):
    assert k != l
    if k + 1 == l:
        return Jp[k]
    if k < l:
        return commutator(Jp[k], E_basis(k + 1, l, Jp, Jm))
    if k == l + 1:
        return Jm[l]
    if k > l:
        return commutator(Jm[k - 1], E_basis(k - 1, l, Jp, Jm))


def generators(S: WEIGHT):
    """
    Returns the generators of the Lie algebra
    """
    N = len(S)
    if N == 1:
        return np.zeros((1, 1, 1))

    Jp = upper_ladder_matrices(S)
    Jm = lower_ladder_matrices(S)
    Jz = Jz_matrices(S)

    Xi = np.stack(
        [
            1j * (E_basis(k, l, Jp, Jm) + E_basis(l, k, Jp, Jm))
            for k in range(N)
            for l in range(k + 1, N)
        ],
        axis=0,
    )
    Xr = np.stack(
        [E_basis(k, l, Jp, Jm) - E_basis(l, k, Jp, Jm) for k in range(N) for l in range(k + 1, N)],
        axis=0,
    )
    Xd = 2j * Jz
    return np.concatenate([Xi, Xr, Xd], axis=0)


def algebra(S: WEIGHT):
    """
    Returns the Lie algebra
    """
    N = len(S)
    X = generators(S)

    x = X.reshape(N**2 - 1, -1)
    pix = np.linalg.pinv(x)
    # np.testing.assert_allclose(x @ pix, np.eye(N**2 - 1), atol=1e-10)

    C = np.einsum("iab,jbc->ijac", X, X) - np.einsum("jab,ibc->ijac", X, X)
    C = C.reshape(N**2 - 1, N**2 - 1, -1)

    A = np.einsum("ijz,zk->ijk", C, pix)
    A = np.real(A)
    A = round_to_sqrt_rational(A)
    return A


@dataclass(frozen=True)
class SUN(TabulatedIrrep):
    S: Tuple[int]  # List of weights of the representation

    def __post_init__(rep):
        assert isinstance(rep.S, tuple)
        _assert_valid_S(rep.S)
        assert rep.S[-1] == 0

    @classmethod
    def from_string(cls, s: str) -> "SUN":
        # (4,3,2,1,0)
        m = re.match(r"\((\d+(?:,\d+)*)\)", s)
        return cls(S=tuple(map(int, m.group(1).split(","))))

    def __mul__(rep1: "SUN", rep2: "SUN") -> List["SUN"]:
        return map(SUN, sorted(set(mul_rep(rep1.S, rep2.S))))

    @classmethod
    def clebsch_gordan(cls, rep1: "SUN", rep2: "SUN", rep3: "SUN") -> np.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        return clebsch_gordan_matrix(rep1.S, rep2.S, rep3.S)

    def __lt__(rep1: "SUN", rep2: "SUN") -> bool:
        return rep1.S < rep2.S

    def __eq__(rep1: "SUN", rep2: "SUN") -> bool:
        return rep1.S == rep2.S

    @property
    def dim(rep: "SUN") -> int:
        return dim(rep.S)

    def is_scalar(rep: "SUN") -> bool:
        """Equivalent to ``S=(0,...,0)``"""
        return all(s == 0 for s in rep.S)

    def discrete_generators(rep: "SUN") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SUN") -> np.ndarray:
        return generators(rep.S)

    def algebra(rep: "SUN") -> np.ndarray:
        if rep.S == (0,):
            return np.zeros((1, 1, 1))
        return algebra((1,) + (0,) * (len(rep.S) - 1))


@dataclass(frozen=True)
class SU2_(SUN):
    def __post_init__(rep):
        assert len(rep.S) == 2

    @classmethod
    def iterator(cls) -> Iterator["SU2_"]:
        for j in itertools.count(0):
            yield SU2_(S=(j, 0))


@dataclass(frozen=True)
class SU3(SUN):
    def __post_init__(rep):
        assert len(rep.S) == 3

    @classmethod
    def iterator(cls) -> Iterator["SU3"]:
        for j1 in itertools.count(0):
            for j2 in range(j1 + 1):
                yield SU3(S=(j1, j2, 0))


@dataclass(frozen=True)
class SU4(SUN):
    def __post_init__(rep):
        assert len(rep.S) == 4

    @classmethod
    def iterator(cls) -> Iterator["SU4"]:
        for j1 in itertools.count(0):
            for j2 in range(j1 + 1):
                for j3 in range(j2 + 1):
                    yield SU4(S=(j1, j2, j3, 0))
