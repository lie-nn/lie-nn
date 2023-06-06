import numpy as np

# import math
import lie_nn as lie
from typing import Tuple


def _num_transpositions(n: int):
    return n * (n - 1) // 2


def _perm_matrix(p: Tuple[int, ...]) -> np.ndarray:
    n = len(p)
    m = np.zeros([n, n])
    m[range(n), p] = 1
    return m


def _transposition_matrix(n: int, i: int, j: int) -> np.ndarray:
    assert 0 <= i < n
    assert 0 <= j < n
    m = np.eye(n)
    m[i, i] = 0
    m[j, j] = 0
    m[i, j] = 1
    m[j, i] = 1
    return m


class Sn_natural(lie.Rep):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def algebra(self) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        return np.zeros((0, 0, 0))

    def continuous_generators(self) -> np.ndarray:
        return np.zeros((0, self.n, self.n))

    def discrete_generators(self) -> np.ndarray:
        H = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                H.append(_transposition_matrix(self.n, i, j))
        return np.stack(H)

    def create_trivial(self) -> lie.GenericRep:
        num = _num_transpositions(self.n)
        return lie.GenericRep(A=self.A, X=np.zeros((0, 1, 1)), H=np.ones((num, 1, 1)))


class Sn_trivial(lie.Rep):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

    def algebra(self) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        return np.zeros((0, 0, 0))

    def continuous_generators(self) -> np.ndarray:
        return np.zeros((0, 1, 1))

    def discrete_generators(self) -> np.ndarray:
        num = _num_transpositions(self.n)
        return np.ones((num, 1, 1))

    def create_trivial(self) -> lie.GenericRep:
        return self
