import numpy as np
from multipledispatch import dispatch

from .infer_change_of_basis import infer_change_of_basis
from .irrep import TabulatedIrrep
from .rep import Rep
from .tensor_product import tensor_product


@dispatch(Rep, Rep, Rep)
def clebsch_gordan(rep1: Rep, rep2: Rep, rep3: Rep, *, round_fn=lambda x: x) -> np.ndarray:
    r"""Computes the Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).

    Args:
        rep1: The first input representation.
        rep2: The second input representation.
        rep3: The output representation.

    Returns:
        The Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).
        It is an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``.
    """
    tp = tensor_product(rep1, rep2)
    cg = infer_change_of_basis(tp, rep3, round_fn=round_fn)
    cg = cg.reshape((-1, rep3.dim, rep1.dim, rep2.dim))
    cg = np.moveaxis(cg, 1, 3)
    return cg


@dispatch(TabulatedIrrep, TabulatedIrrep, TabulatedIrrep)
def clebsch_gordan(  # noqa: F811
    rep1: TabulatedIrrep, rep2: TabulatedIrrep, rep3: TabulatedIrrep, *, round_fn=lambda x: x
) -> np.ndarray:
    return rep1.clebsch_gordan(rep1, rep2, rep3)
