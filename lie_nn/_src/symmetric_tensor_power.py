import itertools

import numpy as np
from multimethod import multimethod

from .conjugate import conjugate
from .change_basis import change_basis
from .rep import ConjRep, MulRep, QRep, Rep, SumRep
from .util import kron, permutation_base, permutation_base_to_matrix


def _symmetric_perm_repr(n: int):
    return frozenset((1, p) for p in itertools.permutations(range(n)))


@multimethod
def symmetric_tensor_power(rep: QRep, n: int) -> Rep:  # noqa: F811
    # out = P @ Q @ tensorpower(rep, n) @ Q^-1 @ P^T
    Q = kron(*[rep.Q] * n)  # [d**n, d**n]
    base = permutation_base(_symmetric_perm_repr(n), (rep.dim,) * n)
    P = permutation_base_to_matrix(base, (rep.dim,) * n)  # [symmetric, d, d, ... d]
    P = np.reshape(P, (len(base), -1))  # [symmetric, d**n]
    S = P @ Q @ P.T

    # out = S @ P @ tensorpower(rep, n) @ P^T @ S^-1
    return change_basis(S, symmetric_tensor_power(rep.rep, n))


@multimethod
def symmetric_tensor_power(rep: SumRep, n: int) -> Rep:  # noqa: F811
    # for all subreps in rep.reps
    # run symmetric_tensor_power(subrep, i) for i = 1, ..., n
    raise NotImplementedError


@multimethod
def symmetric_tensor_power(rep: MulRep, n: int) -> Rep:  # noqa: F811
    # stp = [symmetric_tensor_power(rep.rep, i) for i in range(0, n + 1)]
    # i j if rep.mul == 2 and n == 3
    # 3 0
    # 2 1
    # 1 2
    # 0 3
    raise NotImplementedError


@multimethod
def symmetric_tensor_power(rep: ConjRep, n: int) -> Rep:  # noqa: F811
    return conjugate(symmetric_tensor_power(rep.rep, n))
