import numpy as np

# import math
import lie_nn as lie
from typing import Tuple


def _num_transpositions(n: int):
    return n * (n - 1) // 2


def _permutation_matrix(p: Tuple[int, ...]) -> np.ndarray:
    """
    p = (i j k l ...)

    D[p] e_0 = e_i
    D[p] e_1 = e_j
    """
    n = len(p)
    m = np.zeros([n, n])
    m[p, range(n)] = 1
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
        assert n >= 1
        self.n = n

    def algebra(self) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        return np.zeros((0, 0, 0))

    def continuous_generators(self) -> np.ndarray:
        return np.zeros((0, self.n, self.n))

    def discrete_generators(self) -> np.ndarray:
        if self.n == 1:
            return np.zeros((0, 1, 1))
        if self.n == 2:
            return _transposition_matrix(2, 0, 1)[None, :, :]
        if self.n >= 3:
            return np.stack(
                [
                    _transposition_matrix(self.n, 0, 1),
                    _permutation_matrix(tuple((i + 1) % self.n for i in range(self.n))),
                ]
            )

    def create_trivial(self) -> lie.GenericRep:
        return lie.GenericRep(A=self.A, X=np.zeros((0, 1, 1)), H=np.ones((len(self.H), 1, 1)))


class Sn_standard(lie.Rep):
    """Standard representation of S(n)

    Basis for S(m+1):
        f1 = e1 - e0
        f2 = e2 - e0
        f3 = e3 - e0
        fm = em - e0

    Transposition (0, 1) is represented by:
        D f1 = e0 - e1 = -f1
        D f2 = e2 - e1 = f2 - f1
        D f3 = e3 - e1 = f3 - f1
        D fm = em - e1 = fm - f1

    Cycle (1, 2, 3 ... m, 0) is represented by:
        D f1 = e2 - e1 = f2 - f1
        D f2 = e3 - e1 = f3 - f1
        D f3 = e4 - e1 = f4 - f1
        D fm = e0 - e1 = -f1
    """

    def __init__(self, n) -> None:
        super().__init__()
        assert n >= 2
        self.n = n

    def algebra(self) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        return np.zeros((0, 0, 0))

    def continuous_generators(self) -> np.ndarray:
        return np.zeros((0, self.n - 1, self.n - 1))

    def discrete_generators(self) -> np.ndarray:
        if self.n == 2:
            return -np.ones((1, 1, 1))
        if self.n >= 3:
            H_tr = np.zeros((self.n - 1, self.n - 1))
            H_tr[0, 0] = -1
            for i in range(1, self.n - 1):
                H_tr[i, i] = 1
                H_tr[0, i] = -1

            H_cy = np.zeros((self.n - 1, self.n - 1))
            for i in range(0, self.n - 2):
                H_cy[i + 1, i] = 1
                H_cy[0, i] = -1
            H_cy[0, self.n - 2] = -1
            return np.stack([H_tr, H_cy])

    def create_trivial(self) -> lie.GenericRep:
        return lie.GenericRep(A=self.A, X=np.zeros((0, 1, 1)), H=np.ones((len(self.H), 1, 1)))


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
        if self.n == 1:
            return np.ones((0, 1, 1))
        if self.n == 2:
            return np.ones((1, 1, 1))
        if self.n >= 3:
            return np.ones((2, 1, 1))

    def create_trivial(self) -> lie.GenericRep:
        return self
