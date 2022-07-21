import itertools
from fractions import Fraction
from operator import add
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .. import Irrep, static_jax_pytree
from ..util import round_to_sqrt_rational

WEIGHT = Tuple[int, ...]
GT_PATTERN = Tuple[WEIGHT, ...]


def _assert_valid_S(S: WEIGHT):
    """
    (3, 3, 2) is a valid S.
    (3, 2, 3) is not a valid S.
    """
    assert len(S) > 0
    assert all(s1 >= s2 for s1, s2 in zip(S, S[1:]))


def dim(S: WEIGHT) -> int:
    """Return the number of possible GT-patterns with first line S."""
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
    [(3, 3, 1), (3, 3, 0), (3, 2, 1), (3, 2, 0), (3, 1, 1), (3, 1, 0)]
    """
    _assert_valid_S(S)
    if len(S) == 1:
        return
    diff = [s1 - s2 for s1, s2 in zip(S, S[1:])]
    for x in itertools.product(*[range(s + 1) for s in diff]):
        yield tuple(S[i] - x[i] for i in range(len(S) - 1))


def S_to_Ms(S: WEIGHT) -> Iterator[GT_PATTERN]:
    """Produce all possible GT-patterns with first line S.

    >>> list(S_to_Ms((2, 0)))
    [((2, 0), (2,)), ((2, 0), (1,)), ((2, 0), (0,))]
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
    n = len(M)
    for i in range(1, n):
        for j in range(n - i):
            yield (M[i - 1][j], M[i - 1][j + 1], M[i][j])


def is_valid_M(M: GT_PATTERN):
    return all(m1 >= m3 >= m2 >= 0 for m1, m2, m3 in triangular_triplet(M))


def unique_pairs(n: int, start: int = 0) -> Iterator[Tuple[int, int]]:
    """Produce pairs of indexes in range(n)"""
    for i in range(start, n):
        for j in range(n - i):
            yield i, j


def M_to_sigma(M: GT_PATTERN) -> WEIGHT:
    sigma = [sum(row) for row in M] + [0]
    return sigma[::-1]


def M_to_z_weight(M: GT_PATTERN) -> WEIGHT:
    """The pattern weight of a GT-pattern."""
    w = M_to_p_weight(M)
    return tuple((s2 - s1) / 2 for s1, s2 in zip(w[1:], w))


def M_to_z_weights(Ms: List[GT_PATTERN]) -> List[WEIGHT]:
    return [M_to_z_weight(M) for M in Ms]


def M_to_p_weight(M: GT_PATTERN) -> WEIGHT:
    """The pattern weight of a GT-pattern."""
    sigma = M_to_sigma(M)
    return tuple(s2 - s1 for s1, s2 in zip(sigma, sigma[1:]))


def Ms_to_p_weight(Ms: List[GT_PATTERN]) -> List[WEIGHT]:
    return [M_to_p_weight(M) for M in Ms]


def compute_coeff_lower(M: GT_PATTERN, k, l) -> float:
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
    return (-num / den) ** 0.5


def compute_coeff_upper(M: GT_PATTERN, k, l) -> float:
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
    assert (-num / den) >= 0
    return (-num / den) ** 0.5


def M_add_at_kl(M: GT_PATTERN, k: int, l: int, increment: int) -> Optional[GT_PATTERN]:
    M = tuple(tuple(M[i][j] + increment if (i, j) == (k, l) else M[i][j] for j in range(len(M[i]))) for i in range(len(M)))
    return M if is_valid_M(M) else None


def lower_ladder(M: GT_PATTERN) -> List[Tuple[float, GT_PATTERN]]:
    n = len(M)
    instructions = []
    for k, l in unique_pairs(n, 1):
        M_kl = M_add_at_kl(M, k, l, -1)
        if M_kl is not None:
            coeff = compute_coeff_lower(M, k, l)
            if coeff != 0:
                instructions.append((coeff, M_kl))
    return instructions


def upper_ladder(M: GT_PATTERN) -> List[Tuple[float, GT_PATTERN]]:
    n = len(M)
    instructions = []
    for k, l in unique_pairs(n, 1):
        M_kl = M_add_at_kl(M, k, l, 1)
        if M_kl is not None:
            coeff = compute_coeff_upper(M, k, l)
            if coeff != 0:
                instructions.append((coeff, M_kl, k - 1))
        else:
            instructions.append((0.0, M_kl, k - 1))
    return instructions


def construct_highest_weight_constraint(rep1: "SURep", rep2: "SURep", M_eldest: GT_PATTERN) -> np.ndarray:
    n = len(M_eldest)
    A_1 = np.zeros((rep1.dim, rep2.dim, n - 1), dtype=np.float64)
    A_list = []
    for i in range(rep1.dim):
        for j in range(rep2.dim):
            M_1 = index_to_M(rep1.S, i)
            M_2 = index_to_M(rep2.S, j)
            W_1 = M_to_z_weight(M_1)
            W_2 = M_to_z_weight(M_2)
            W_3 = M_to_z_weight(M_eldest)
            if tuple(map(add, W_1, W_2)) == W_3:
                for (instruction, instruction_p) in zip(upper_ladder(M_1), upper_ladder(M_2)):
                    if instruction[1] is not None:
                        l_dim, coeff = instruction[2], instruction[0]
                        A_1[i, j, l_dim] += coeff
                    if instruction_p[1] is not None:
                        l_dim, coeff = instruction_p[2], instruction_p[0]
                        A_1[i, j, l_dim] += coeff
            else:
                A = np.zeros((rep1.dim, rep2.dim, 1))
                A[i, j, :] += 1
                A_list.append(A)
    return round_to_sqrt_rational(np.concatenate((A_1, *A_list), axis=-1))


@static_jax_pytree
class SURep(Irrep):
    S: Tuple[int]  # List of weights of the representation

    def __mul__(rep1: "SURep", rep2: "SURep") -> List["SURep"]:
        n = len(rep2.S)
        for pattern in S_to_Ms(rep1.S):
            t_weight = list(rep2.S)
            for l, k in unique_pairs(n):
                try:
                    t_weight[n - k - 1] += pattern[k][l] - pattern[k + 1][l]
                except IndexError:
                    t_weight[n - k - 1] += pattern[k][l]
                if n - k - 1 >= 1 and t_weight[n - k - 1] > t_weight[n - k - 2]:
                    t_weight = []
                    break
            if t_weight:
                yield SURep(S=tuple(x - t_weight[-1] for x in t_weight))

    @classmethod
    def clebsch_gordan(cls, rep1: "SURep", rep2: "SURep", rep3: "SURep") -> np.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        pass

    @property
    def dim(rep: "SURep") -> int:
        # A numerical algorithm for the explicit calculation of SU(N) and SL(N, C)
        # Clebsch-Gordan coefficients Arne Alex, Matthias Kalus, Alan Huckleberry
        # and Jan von Delft Eq 22.
        return dim(rep.S)

    @classmethod
    def iterator(self, cls) -> Iterator["SURep"]:
        pass

    def discrete_generators(rep: "SURep") -> np.ndarray:
        return np.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: "SURep") -> np.ndarray:
        pass

    @classmethod
    def algebra(self, cls) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        lie_algebra_real = np.zeros((self.n**2 - 1, self.n, self.n))
        lie_algebra_imag = np.zeros((self.n**2 - 1, self.n, self.n))
        k = 0
        for i in range(self.n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k, i, j] = 1
                lie_algebra_real[k, j, i] = -1
                k += 1
                # symmetric imaginary generators
                lie_algebra_imag[k, i, j] = 1
                lie_algebra_imag[k, j, i] = 1
                k += 1
        for i in range(self.n - 1):
            # diagonal traceless imaginary generators
            lie_algebra_imag[k, i, i] = 1
            for j in range(self.n):
                if i == j:
                    continue
                lie_algebra_imag[k, j, j] = -1 / (self.n - 1)
            k += 1
        return lie_algebra_real + lie_algebra_imag * 1j
