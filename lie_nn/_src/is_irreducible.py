import numpy as np
from multipledispatch import dispatch

from .rep import Irrep, Rep
from .util import is_irreducible as _is_irreducible


@dispatch(Rep, epsilon=object)
def is_irreducible(rep: Rep, *, epsilon: float = 1e-10) -> bool:
    """Returns True if the representation is irreducible."""
    return _is_irreducible(np.concatenate([rep.X, rep.H], axis=0), epsilon=epsilon)


@dispatch(Irrep, epsilon=object)
def is_irreducible(rep: Irrep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return True
