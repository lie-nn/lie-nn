import numpy as np
from multimethod import multimethod

from .change_basis import change_basis
from .conjugate import conjugate
from .direct_sum import direct_sum
from .infer_change_of_basis import infer_change_of_basis
from .multiply import multiply
from .rep import ConjRep, GenericRep, Irrep, MulRep, QRep, Rep, SumRep
from .util import decompose_rep_into_irreps


@multimethod
def reduce(rep: ConjRep) -> Rep:  # noqa: F811
    return conjugate(reduce(rep.rep))


@multimethod
def reduce(rep: MulRep) -> Rep:  # noqa: F811
    return multiply(rep.mul, reduce(rep.rep))


@multimethod
def reduce(rep: QRep) -> Rep:  # noqa: F811
    return change_basis(rep.Q, reduce(rep.rep))


@multimethod
def reduce(rep: SumRep) -> Rep:  # noqa: F811
    return direct_sum(*[reduce(subrep) for subrep in rep.reps])


@multimethod
def reduce(rep: Irrep) -> Irrep:  # noqa: F811
    return rep


@multimethod
def reduce(rep: Rep) -> Rep:  # noqa: F811
    r"""Reduce an unknown representation to a reduced form.
    This operation is slow and should be avoided if possible.
    """
    Ys = decompose_rep_into_irreps(np.concatenate([rep.X, rep.H]))
    d = rep.lie_dim
    Qs = []
    irs = []
    for mul, Y in Ys:
        ir = GenericRep(rep.A, Y[:d], Y[d:])
        Q = infer_change_of_basis(ir, rep)
        assert len(Q) == mul, (len(Q), mul)
        Q = np.einsum("mij->imj", Q).reshape((rep.dim, mul * ir.dim))
        Qs.append(Q)
        if mul > 1:
            ir = MulRep(mul, ir, force=True)
        irs.append(ir)

    rep = direct_sum(*irs)

    Q = np.concatenate(Qs, axis=1)
    if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
        return rep

    return change_basis(Q, rep)
