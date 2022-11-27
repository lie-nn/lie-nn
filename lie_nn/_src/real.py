import lie_nn as lie
import scipy.linalg as la


def is_real(rep: lie.Rep) -> bool:
    S = lie.infer_change_of_basis(rep, lie.conjugate(rep))
    return S.shape[0] > 0


def make_explicitly_real(rep, round_fn=lambda x: x):
    S = lie.infer_change_of_basis(rep, lie.conjugate(rep), round_fn=round_fn)
    assert S.shape[0] == 1
    return lie.change_basis(rep, la.sqrtm(S[0]))
