import numpy as np
import lie_nn as lie


def test_infer_change_of_basis():
    rep1 = lie.change_basis(lie.irreps.SU2(2), np.random.randn(3, 3))
    rep2 = lie.change_basis(lie.irreps.SU2(2), np.random.randn(3, 3))

    Q = lie.infer_change_of_basis(rep1, rep2)[0]
    rep3 = lie.change_basis(rep1, Q)

    np.testing.assert_allclose(rep2.X, rep3.X)


def test_infer_change_of_basis_generic():
    rep1 = lie.change_basis(lie.irreps.SU2(2), np.random.randn(3, 3))
    rep2 = lie.change_basis(lie.irreps.SU2(2), np.random.randn(3, 3))

    rep1 = lie.GenericRep.from_rep(rep1)
    rep2 = lie.GenericRep.from_rep(rep2)

    Q = lie.infer_change_of_basis(rep1, rep2)[0]
    rep3 = lie.change_basis(rep1, Q)

    np.testing.assert_allclose(rep2.X, rep3.X)
