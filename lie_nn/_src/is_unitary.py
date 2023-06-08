import numpy as np
from multipledispatch import dispatch

from .rep import Rep, SumRep, MulRep


@dispatch(Rep)
def is_unitary(rep: Rep) -> bool:
    X = rep.continuous_generators()
    H = rep.discrete_generators()
    # exp(X) @ exp(X^H) = 1
    # X + X^H = 0
    H_unit = np.allclose(H @ np.conj(np.transpose(H, (0, 2, 1))), np.eye(rep.dim), atol=1e-13)
    X_unit = np.allclose(X + np.conj(np.transpose(X, (0, 2, 1))), 0, atol=1e-13)
    return H_unit and X_unit


@dispatch(SumRep)
def is_unitary(rep: SumRep) -> bool:  # noqa: F811
    return all(is_unitary(subrep) for subrep in rep.reps)


@dispatch(MulRep)
def is_unitary(rep: MulRep) -> bool:  # noqa: F811
    return is_unitary(rep.rep)
