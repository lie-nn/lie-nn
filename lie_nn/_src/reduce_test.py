import numpy as np
import lie_nn as lie


def test_reduce1():
    rep1 = lie.change_basis(np.random.randn(3, 3), lie.irreps.SU2(2))
    rep2 = lie.tensor_product(rep1, rep1)
    rep2 = lie.GenericRep.from_rep(rep2)

    rep3 = lie.reduce(rep2)

    np.testing.assert_allclose(rep2.X, rep3.X, atol=1e-5)


def test_reduce2():
    rep = lie.tensor_product(lie.irreps.SU2(2), lie.irreps.SU2(2))
    Q = np.random.randn(2 * rep.dim, 2 * rep.dim)
    rep1 = lie.change_basis(Q, lie.direct_sum(rep, rep))
    rep2 = lie.GenericRep.from_rep(rep1)

    rep1 = lie.reduce(rep1)
    rep2 = lie.reduce(rep2)

    np.testing.assert_allclose(rep1.X, rep2.X, atol=1e-5)
