import numpy as np
import lie_nn as lie


def test_infer_change_of_basis():
    rep1 = lie.change_basis(np.random.randn(3, 3), lie.irreps.SU2(2))
    rep2 = lie.change_basis(np.random.randn(3, 3), lie.irreps.SU2(2))

    Q = lie.infer_change_of_basis(rep1, rep2)[0]
    rep3 = lie.change_basis(Q, rep1)

    np.testing.assert_allclose(rep2.X, rep3.X)


def test_infer_change_of_basis_generic():
    rep1 = lie.change_basis(np.random.randn(3, 3), lie.irreps.SU2(2))
    rep2 = lie.change_basis(np.random.randn(3, 3), lie.irreps.SU2(2))

    rep1 = lie.GenericRep.from_rep(rep1)
    rep2 = lie.GenericRep.from_rep(rep2)

    Q = lie.infer_change_of_basis(rep1, rep2)[0]
    rep3 = lie.change_basis(Q, rep1)

    np.testing.assert_allclose(rep2.X, rep3.X)
