import itertools
from fractions import Fraction
from typing import Iterator, List, Tuple

import jax.numpy as jnp

from ._abstract_rep import AbstractRep, static_jax_pytree


def _assert_valid_S(S: Tuple[int, ...]):
    """
    (3, 3, 2) is a valid S.
    (3, 2, 3) is not a valid S.
    """
    assert len(S) > 0
    assert all(s1 >= s2 for s1, s2 in zip(S, S[1:]))


def dim(S: Tuple[int, ...]) -> int:
    """Return the number of possible GT-patterns with first line S."""
    _assert_valid_S(S)
    d = 1
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d *= 1 + Fraction(S[i] - S[j], j - i)
    assert d.denominator == 1
    return int(d)


assert dim((3, 3, 1, 0)) == sum(dim((3, a, b)) for a in [3, 2, 1] for b in [1, 0])


def S_to_Ss(S: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
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


def S_to_Ms(S: Tuple[int, ...]) -> Iterator[Tuple[Tuple[int, ...], ...]]:
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


def index_to_M(S: Tuple[int, ...], index: int) -> Tuple[Tuple[int, ...], ...]:
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


def M_to_index(M: Tuple[Tuple[int, ...], ...]) -> int:
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


def triangular_triplet(M: Tuple[Tuple[int, ...], ...]):
    n = len(M)
    for i in range(1, n):
        for j in range(n - i):
            yield (M[i - 1][j], M[i - 1][j + 1], M[i][j])


def _assert_valid_M(M: Tuple[Tuple[int, ...], ...]):
    if all(m1 >= m3 >= m2 for m1, m2, m3 in triangular_triplet(M)):
        return True
    else:
        return False


def unique_pairs(n: int):
    """Produce pairs of indexes in range(n)"""
    for i in range(n):
        for j in range(n - i):
            yield i, j


def M_to_sigma(M: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
    sigma = [sum(row) for row in M] + [0]
    return sigma[::-1]


def Ms_to_p_weight(Ms: List[Tuple[Tuple[int, ...], ...]]) -> Tuple[Tuple[int, ...], ...]:
    p_weights = []
    for M in Ms:
        sigma = M_to_sigma(M)
        p_weight = tuple(s2 - s1 for s1, s2 in zip(sigma, sigma[1:]))
        p_weights.append(p_weight)
    return tuple(p_weights)


def compute_coff_lower(M: Tuple[Tuple[int, ...], ...], k, l) -> float:
    num_1 = 1
    for k_p in range(l + 1):
        num_1 *= M[l + 1][k_p] - M[l][k] + k - k_p + 1
    num_2 = 1
    for k_p in range(l - 1):
        num_1 *= M[l - 1][k_p] - M[l][k] + k - k_p
    den = 1
    for k_p in range(l - 1) and k_p != k:
        den *= (M[l][k_p] - M[l][k] + k - k_p + 1) * (M[l][k_p] - M[l][k] + k - k_p)
    return (-(num_1 * num_2) / den) ** 0.5


def lower_ladder(M: Tuple[Tuple[int, ...], ...]) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
    instructions = []
    for l, k in unique_pairs(len(M)):
        M_kl = list(M)
        M_kl[k][l] -= 1
        if not _assert_valid_M(M_kl):
            continue
        else:
            coeff = compute_coff_lower(M, k, l)
        instructions.append((coeff, M_kl))


@static_jax_pytree
class SURep(AbstractRep):
    S: Tuple[int]  # List of weights of the representation

    def __mul__(rep1: 'SURep', rep2: 'SURep') -> List['SURep']:
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
    def clebsch_gordan(cls, rep1: 'SURep', rep2: 'SURep', rep3: 'SURep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        pass

    @property
    def dim(self, rep: 'SURep') -> int:
        # A numerical algorithm for the explicit calculation of SU(N) and SL(N, C)
        # Clebsch-Gordan coefficients Arne Alex, Matthias Kalus, Alan Huckleberry
        # and Jan von Delft Eq 22.
        return dim(rep.S)

    @classmethod
    def iterator(self, cls) -> Iterator['SURep']:
        pass

    def discrete_generators(rep: 'SURep') -> jnp.ndarray:
        return jnp.zeros((0, rep.dim, rep.dim))

    def continuous_generators(rep: 'SURep') -> jnp.ndarray:
        pass

    @classmethod
    def algebra(self, cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        lie_algebra_real = jnp.zeros((self.n**2 - 1, self.n, self.n))
        lie_algebra_imag = jnp.zeros((self.n**2 - 1, self.n, self.n))
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
