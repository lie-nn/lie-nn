import numpy as np
from multimethod import multimethod

import lie_nn as lie


@multimethod
def reduce(rep: lie.ConjRep) -> lie.Rep:  # noqa: F811
    return lie.conjugate(reduce(rep.rep))


@multimethod
def reduce(rep: lie.MulRep) -> lie.Rep:  # noqa: F811
    return lie.multiply(rep.mul, reduce(rep.rep))


@multimethod
def reduce(rep: lie.QRep) -> lie.Rep:  # noqa: F811
    return lie.change_basis(rep.Q, reduce(rep.rep))


@multimethod
def reduce(rep: lie.SumRep) -> lie.Rep:  # noqa: F811
    return lie.direct_sum(*[reduce(subrep) for subrep in rep.reps])


@multimethod
def reduce(rep: lie.Irrep) -> lie.Irrep:  # noqa: F811
    return rep


@multimethod
def reduce(rep: lie.Rep) -> lie.Rep:  # noqa: F811
    r"""Reduce an unknown representation to a reduced form.
    This operation is slow and should be avoided if possible.
    """
    Ys = lie.utils.decompose_rep_into_irreps(np.concatenate([rep.X, rep.H]))
    d = rep.lie_dim
    Qs = []
    irs = []
    for mul, Y in Ys:
        ir = lie.GenericRep(rep.A, Y[:d], Y[d:])
        Q = lie.infer_change_of_basis(ir, rep)
        assert len(Q) == mul, (len(Q), mul)
        Q = np.einsum("mij->imj", Q).reshape((rep.dim, mul * ir.dim))
        Qs.append(Q)
        if mul > 1:
            ir = lie.MulRep(mul, ir, force=True)
        irs.append(ir)

    rep = lie.direct_sum(*irs)

    Q = np.concatenate(Qs, axis=1)
    if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
        return rep

    return lie.change_basis(Q, rep)
