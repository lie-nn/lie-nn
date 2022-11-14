import numpy as np
from multipledispatch import dispatch

from .irrep import Irrep
from .rep import Rep
from .util import infer_change_of_basis, kron, vmap

# TODO(mario): create a function infer_change_of_basis(Rep, Rep) -> np.ndarray
# TODO(mario): use tensor_product and infer_change_of_basis to implement clebsch_gordan


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
    # Check the group structure
    assert np.allclose(rep1.algebra(), rep2.algebra())
    assert np.allclose(rep2.algebra(), rep3.algebra())

    i1 = np.eye(rep1.dim)
    i2 = np.eye(rep2.dim)

    X_in = vmap(lambda x1, x2: kron(x1.T, i2) + kron(i1, x2.T))(rep1.continuous_generators(), rep2.continuous_generators())
    X_out = vmap(lambda x3: x3.T)(rep3.continuous_generators())

    H_in = vmap(lambda x1, x2: kron(x1.T, x2.T), out_shape=(rep1.dim * rep2.dim, rep1.dim * rep2.dim))(
        rep1.discrete_generators(), rep2.discrete_generators()
    )
    H_out = vmap(lambda x3: x3.T, out_shape=(rep3.dim, rep3.dim))(rep3.discrete_generators())

    Y_in = np.concatenate([X_in, H_in])
    Y_out = np.concatenate([X_out, H_out])
    cg = infer_change_of_basis(Y_in, Y_out, round_fn=round_fn)
    np.testing.assert_allclose(
        np.einsum("aij,bjk->abik", Y_in, cg),
        np.einsum("bij,ajk->abik", cg, Y_out),
        rtol=1e-10,
        atol=1e-10,
    )

    assert cg.dtype in [np.float64, np.complex128], "Clebsch-Gordan coefficient must be computed with double precision."

    cg = round_fn(cg * np.sqrt(rep3.dim))
    cg = cg.reshape((-1, rep1.dim, rep2.dim, rep3.dim))
    return cg


@dispatch(Irrep, Irrep, Irrep)
def clebsch_gordan(rep1: Irrep, rep2: Irrep, rep3: Irrep) -> np.ndarray:
    return rep1.clebsch_gordan(rep1, rep2, rep3)
