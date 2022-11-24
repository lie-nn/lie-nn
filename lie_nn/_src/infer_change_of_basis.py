import numpy as np
from multipledispatch import dispatch

from .rep import GenericRep, Rep
from .util import infer_change_of_basis as _infer_change_of_basis


# @dispatch(Rep, Rep, object)
def infer_change_of_basis(rep1: Rep, rep2: Rep, round_fn=lambda x: x) -> np.ndarray:
    # Check the group structure
    assert np.allclose(rep1.algebra(), rep2.algebra())

    Y1 = np.concatenate([rep1.continuous_generators(), rep1.discrete_generators()])
    Y2 = np.concatenate([rep2.continuous_generators(), rep2.discrete_generators()])

    A = _infer_change_of_basis(Y2, Y1, round_fn=round_fn)
    np.testing.assert_allclose(
        np.einsum("aij,bjk->abik", Y2, A), np.einsum("bij,ajk->abik", A, Y1), rtol=1e-10, atol=1e-10,
    )

    assert A.dtype in [np.float64, np.complex128]

    A = A * np.sqrt(rep2.dim)
    A = round_fn(A)
    return A
