import numpy as np
import scipy.linalg as la

import lie_nn as lie


def is_real(rep: lie.Rep) -> bool:
    S = lie.infer_change_of_basis(rep, lie.conjugate(rep))
    return S.shape[0] > 0


def make_explicitly_real(rep: lie.Rep, *, round_fn=lambda x: x):
    S = lie.infer_change_of_basis(rep, lie.conjugate(rep), round_fn=round_fn)
    if S.shape[0] == 0:
        raise ValueError("The representation is not real")
    assert S.shape[0] == 1
    rep = lie.change_basis(rep, la.sqrtm(S[0]))
    if np.linalg.norm(rep.X.imag) > 1e-10:
        raise ValueError("The representation is not real")
    return rep
