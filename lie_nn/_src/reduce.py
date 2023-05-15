import numpy as np
from multipledispatch import dispatch

from .reduced_rep import MulIrrep, ReducedRep
from .rep import GenericRep
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
    return rep


@dispatch(GenericRep)
def reduce(rep: GenericRep) -> ReducedRep:  # noqa: F811
    r"""Reduce an unknown representation to a reduced form.
    This operation is slow and should be avoided if possible.
    """
    Ys = decompose_rep_into_irreps(np.stack([rep.X, rep.H], axis=0))
    # TODO: change of basis
    return ReducedRep(rep.A, tuple(MulIrrep(1, Y) for Y in Ys))
