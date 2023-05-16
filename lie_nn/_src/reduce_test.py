import numpy as np
import lie_nn as lie


def test_reduce():
    rep1 = lie.change_basis(lie.irreps.SU2(2), np.random.randn(3, 3))
    rep2 = lie.tensor_product(rep1, rep1)
    rep2 = lie.GenericRep.from_rep(rep2)

    rep3 = lie.reduce(rep2)

    np.testing.assert_allclose(rep2.X, rep3.X, atol=1e-5)
