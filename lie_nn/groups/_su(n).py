import itertools
from typing import Iterator, List

from flax import struct
import jax.numpy as jnp

from ._abstract_rep import AbstractRep


def get_egein_value(l: int, lls: List[int]):
    np_lls = jnp.array(lls)
    row_sum_lp1 = jnp.cumsum(np_lls[:l + 1])
    row_sum_lm1 = jnp.cumsum(np_lls[:l - 1])
    row_sum_l = jnp.cumsum(np_lls[:l])
    return row_sum_l - (1 / 2) * (row_sum_lm1 + row_sum_lp1)


def branching(weight: list[int], depth: int, pattern: dict):
    weights = []
    for node in range(len(weight) - 1):
        snake_values = []
        for j in range(weight[node] - weight[node + 1] + 1):
            snake_values.append(j)
        pattern[node, depth] = snake_values
        weights.append(snake_values)
    return weights


@struct.dataclass
class SURep(AbstractRep):
    n: int = struct.field(pytree_node=False)  # dimension of the SU(n) group
    lls: List[int] = struct.field(pytree_node=False)  # List of weights of the representation

    def __init__(self, n):
        self.n = n

    def z_weight(rep: 'SURep'):
        return [get_egein_value(l, rep.lls) for l in range(rep.n)]

    def GT_patterns(rep: 'SURep'):
        pattern = {}
        n = rep.n
        for i in range(n):
            if i == 0:
                rep = branching(rep, 1)
        else:
            for weights in tuple(itertools.product(*rep)):
                rep = branching(weights, i + 1)
        return pattern

    def __mul__(rep1: 'SURep', rep2: 'SURep') -> List['SURep']:
        assert rep1.n == rep1.n
        n = rep1.n
        l = rep1.n
        patterns = rep1.GT_patterns()
        condition = True
        for k in range(1, n + 1):
            l -= 1
            if k <= l:
                pass

    @classmethod
    def clebsch_gordan(cls, rep1: 'SURep', rep2: 'SURep', rep3: 'SURep') -> jnp.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        pass

    @property
    def dim(self, rep: 'SURep') -> int:
        # A numerical algorithm for the explicit calculation of SU(N) and SL(N, C)
        # Clebsch-Gordan coefficients Arne Alex, Matthias Kalus, Alan Huckleberry
        # and Jan von Delft Eq 22.
        numerator = 1
        denomiator = 1
        for i in range(self.n):
            for sum in range(self.n):
                j = sum - i
                numerator *= rep.lls[j] - rep.lls[i + j] + i
                denomiator *= i
        i += i

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
