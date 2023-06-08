import numpy as np
from multipledispatch import dispatch

from .change_basis import change_basis, project
from .direct_sum import direct_sum
from .multiply import multiply
from .rep import GenericRep, MulRep, QRep, Rep, SumRep, PRep


@dispatch(Rep)
def conjugate(rep: Rep) -> GenericRep:
    return GenericRep(
        A=rep.A,
        X=np.conjugate(rep.X),
        H=np.conjugate(rep.H),
    )


@dispatch(QRep)
def conjugate(rep: QRep) -> Rep:  # noqa: F811
    return change_basis(np.conjugate(rep.Q), conjugate(rep.rep))


@dispatch(PRep)
def conjugate(rep: PRep) -> Rep:  # noqa: F811
    return project(np.conjugate(rep.Q), conjugate(rep.rep))


@dispatch(SumRep)
def conjugate(rep: SumRep) -> Rep:  # noqa: F811
    return direct_sum(*[conjugate(subrep) for subrep in rep.reps])


@dispatch(MulRep)
def conjugate(rep: MulRep) -> Rep:  # noqa: F811
    return multiply(rep.mul, conjugate(rep.rep))
