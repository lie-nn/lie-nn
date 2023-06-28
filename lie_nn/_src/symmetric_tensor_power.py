import itertools

import numpy as np
from multimethod import multimethod

from .conjugate import conjugate
from .change_basis import change_basis
from .rep import ConjRep, MulRep, QRep, Rep, SumRep
from .utils import kron, permutation_base, permutation_base_to_matrix
import lie_nn as lie


def _symmetric_perm_repr(n: int):
    return frozenset((1, p) for p in itertools.permutations(range(n)))


def _symmetric_perm_matrix(d: int, n: int):
    base = permutation_base(_symmetric_perm_repr(n), (d,) * n)
    P = permutation_base_to_matrix(base, (d,) * n)  # [symmetric, d, d, ... d]
    P = np.reshape(P, (P.shape[0], -1))  # [symmetric, d**n]
    return P


@multimethod
def symmetric_tensor_power(rep: QRep, n: int) -> Rep:  # noqa: F811
    # out = P @ Q @ tensorpower(rep, n) @ Q^-1 @ P^T
    Q = kron(*[rep.Q] * n)  # [d**n, d**n]
    P = _symmetric_perm_matrix(rep.dim, n)  # [symmetric, d**n]
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


@multimethod
def symmetric_tensor_power(rep: Rep, n: int) -> Rep:  # noqa: F811
    if n == 1:
        return rep

    a = n // 2
    b = n - a

    Pa = _symmetric_perm_matrix(rep.dim, a)
    Pb = _symmetric_perm_matrix(rep.dim, b)

    Ra = symmetric_tensor_power(rep, a)  # Pa @ tensorpower(rep, n // 2) @ Pa^T
    Rb = symmetric_tensor_power(rep, b)  # Pb @ tensorpower(rep, n - n // 2) @ Pb^T

    Pab = kron(Pa, Pb)
    Rab = lie.reduce(lie.tensor_product(Ra, Rb))  # Pab @ tensorpower(rep, n) @ Pab^T

    Pn = _symmetric_perm_matrix(rep.dim, n)
    S = Pn @ Pab.T  # [sym(n), sym(a) * sym(b)]

    print(f"project {Rab} of dim {Rab.dim} ")
    print(f"with S of dim {S.shape} ")
    print(S)

    # return _project(S, Rab)
    raise NotImplementedError


# @multimethod
# def _project(S: np.ndarray, rep: QRep) -> Rep:  # noqa: F811
#     # Assume S S^T = I but not S^T S = I
#     # out = S @ Q @ rep @ Q^-1 @ S^T
#     # u, s, vh = np.linalg.svd(S @ rep.Q, full_matrices=False)
#     # np.testing.assert_allclose(u @ np.diag(s) @ vh, S @ rep.Q, atol=1e-10)
#     # print(f"u {u.shape}\n{u}")
#     # print(f"s {s.shape}\n{s}")
#     # print(f"vh {vh.shape}\n{vh}")
#     # return change_basis(u @ np.diag(s), _project(vh, rep.rep))
#     P = S @ rep.Q
#     print(f"P {P.shape}\n{P}")
#     return _project(P, rep.rep)


# @multimethod
# def _project(S: np.ndarray, rep: SumRep) -> Rep:  # noqa: F811
#     # Assume each subrep is independently projected
#     i = 0
#     reps = []
#     Qs = []
#     for subrep in rep.reps:
#         j = i + subrep.dim
#         Sij = S[:, i:j]
#         i = j

#         print(f"subrep={subrep}, Sij {Sij.shape}\n{Sij}")

#         if np.linalg.norm(Sij) > 1e-10:
#             reps.append(subrep)
#             Qs.append(Sij)

#     Q = np.concatenate(Qs, axis=1)
#     return lie.change_basis(Q, lie.direct_sum(*reps))


# @multimethod
# def _project(S: np.ndarray, rep: MulRep) -> Rep:  # noqa: F811
#     # Assume rep.rep is an irrep
#     S = np.reshape(S, (S.shape[0], rep.mul, rep.dim))
#     x = S[:, :, 0].T  # [mul, d_out]
#     _, u = lie.util.gram_schmidt_with_change_of_basis(x)
#     mul = u.shape[0]  # new mul
#     S = np.einsum("uv , dvi -> dui", u, S)  # [d_out, mul', d]
#     S = np.reshape(S, (S.shape[0], -1))  # [d_out, mul' * d]
#     assert S.shape[0] == S.shape[1], S.shape
#     return lie.change_basis(S, lie.MulRep(mul, rep.rep, force=True))


# @multimethod
# def _project(S: np.ndarray, rep: Rep):  # noqa: F811
#     assert S.shape[0] == S.shape[1], (S, rep.dim)
#     return lie.change_basis(S, rep)
