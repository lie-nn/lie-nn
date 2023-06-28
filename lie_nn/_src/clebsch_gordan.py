import numpy as np
from multimethod import multimethod


import lie_nn as lie


@multimethod
def clebsch_gordan(
    rep1: lie.Rep, rep2: lie.Rep, rep3: lie.Rep, *, round_fn=lambda x: x
) -> np.ndarray:
    r"""Computes the Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).

    Args:
        rep1: The first input representation.
        rep2: The second input representation.
        rep3: The output representation.

    Returns:
        The Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).
        It is an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``.
    """
    tp = lie.tensor_product(rep1, rep2)
    cg = lie.infer_change_of_basis(tp, rep3, round_fn=round_fn)
    cg = cg.reshape((-1, rep3.dim, rep1.dim, rep2.dim))
    cg = np.moveaxis(cg, 1, 3)
    return cg


@multimethod
def clebsch_gordan(  # noqa: F811
    rep1: lie.TabulatedIrrep,
    rep2: lie.TabulatedIrrep,
    rep3: lie.TabulatedIrrep,
    *,
    round_fn=lambda x: x,
) -> np.ndarray:
    return rep1.clebsch_gordan(rep1, rep2, rep3)
