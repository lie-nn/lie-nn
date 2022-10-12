from multipledispatch import dispatch

from . import GenericRep, MulIrrep, ReducedRep


@dispatch(MulIrrep)
def reduce(rep: MulIrrep) -> ReducedRep:
    return ReducedRep(
        A=rep.algebra(),
        irreps=(rep,),
        Q=None,
    )


@dispatch(ReducedRep)
def reduce(rep: ReducedRep) -> ReducedRep:
    return rep


@dispatch(GenericRep)
def reduce(rep: GenericRep) -> ReducedRep:
    r"""Reduce an unknown representation to a reduced form.
    This operation is slow and should be avoided if possible.
    """
    raise NotImplementedError
