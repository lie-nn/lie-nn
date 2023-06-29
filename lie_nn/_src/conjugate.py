import numpy as np
from multimethod import multimethod

import lie_nn as lie


@multimethod
def conjugate(rep: lie.Rep) -> lie.GenericRep:
    return lie.ConjRep(rep, force=True)


@multimethod
def conjugate(rep: lie.QRep) -> lie.Rep:  # noqa: F811
    return lie.change_basis(np.conjugate(rep.Q), conjugate(rep.rep))


@multimethod
def conjugate(rep: lie.SumRep) -> lie.Rep:  # noqa: F811
    return lie.direct_sum(*[conjugate(subrep) for subrep in rep.reps])


@multimethod
def conjugate(rep: lie.MulRep) -> lie.Rep:  # noqa: F811
    return lie.multiply(rep.mul, conjugate(rep.rep))
