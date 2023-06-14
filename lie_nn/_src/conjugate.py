import numpy as np
from multimethod import multimethod

from .change_basis import change_basis
from .direct_sum import direct_sum
from .multiply import multiply
from .rep import ConjRep, GenericRep, MulRep, QRep, Rep, SumRep


@multimethod
def conjugate(rep: Rep) -> GenericRep:
    return ConjRep(rep, force=True)


@multimethod
def conjugate(rep: QRep) -> Rep:  # noqa: F811
    return change_basis(np.conjugate(rep.Q), conjugate(rep.rep))


@multimethod
def conjugate(rep: SumRep) -> Rep:  # noqa: F811
    return direct_sum(*[conjugate(subrep) for subrep in rep.reps])


@multimethod
def conjugate(rep: MulRep) -> Rep:  # noqa: F811
    return multiply(rep.mul, conjugate(rep.rep))
