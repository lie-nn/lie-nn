import numpy as np
from multipledispatch import dispatch

from .infer_change_of_basis import infer_change_of_basis
from .reduced_rep import MulIrrep, ReducedRep
from .rep import GenericRep, Rep
from .util import decompose_rep_into_irreps


@dispatch(MulIrrep)
def reduce(rep: MulIrrep) -> ReducedRep:
    return ReducedRep(
        A=rep.algebra(),
        irreps=(rep,),
        Q=None,
    )


@dispatch(ReducedRep)
def reduce(rep: ReducedRep) -> ReducedRep:  # noqa: F811
    # TODO if we change ReducedRep into SumRep, then reduce its constituents
    return rep


@dispatch(Rep)
def reduce(rep: Rep) -> ReducedRep:  # noqa: F811
    r"""Reduce an unknown representation to a reduced form.
    This operation is slow and should be avoided if possible.
    """
    Ys = decompose_rep_into_irreps(np.concatenate([rep.X, rep.H]))
    Ys = sorted(Ys, key=lambda x: x.shape[1])
    d = rep.lie_dim
    Qs = []
    irs = []
    for Y in Ys:
        ir = GenericRep(rep.A, Y[:d], Y[d:])
        Q = infer_change_of_basis(ir, rep)
        Q = np.einsum("mij->imj", Q).reshape((rep.dim, ir.dim))
        Qs.append(Q)
        irs.append(ir)

    Q = np.concatenate(Qs, axis=1)
    rep = ReducedRep(rep.A, tuple(MulIrrep(1, ir) for ir in irs), Q)
    return rep
