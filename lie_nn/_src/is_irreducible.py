import numpy as np

from .rep import Rep
from .util import is_irreducible as _is_irreducible


def is_irreducible(rep: Rep, *, epsilon: float = 1e-10) -> bool:
    """Returns True if the representation is irreducible."""
    return _is_irreducible(np.concatenate([rep.X, rep.H], axis=0), epsilon=epsilon)
