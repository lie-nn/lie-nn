import numpy as np
import lie_nn as lie
import itertools


def test_reduce1():
    Q = np.array([[2.0, 0.0, 1.0], [1.0, -1.0, 0.0], [0.1, 0.1, 0.1]])
    rep1 = lie.change_basis(Q, lie.irreps.SU2(2))
    rep2 = lie.tensor_product(rep1, rep1)
    rep2 = lie.GenericRep.from_rep(rep2)

    rep3 = lie.reduce(rep2)

    np.testing.assert_allclose(rep2.X, rep3.X, atol=1e-5)


def test_reduce2():
    rep = lie.tensor_product(lie.irreps.SU2(2), lie.irreps.SU2(2))
    rep = lie.direct_sum(rep, rep)

    Q = np.random.randn(rep.dim, rep.dim)
    u, s, vh = np.linalg.svd(Q)
    Q = u @ vh

    rep1 = lie.change_basis(Q, rep)

    rep1 = lie.reduce(rep1)
    rep2 = lie.reduce(lie.GenericRep.from_rep(rep1))

    np.testing.assert_allclose(rep1.X, rep2.X, atol=1e-5)


def test_reduce3():
    rep1 = lie.tensor_power(lie.irreps.SU2(1), 4)

    rep2 = lie.reduce(rep1)

    np.testing.assert_allclose(rep1.X, rep2.X, atol=1e-5)

    irs = [ir for mul, ir in rep2.reps]

    for ir1, ir2 in itertools.combinations(irs, 2):
        assert not lie.are_isomorphic(ir1, ir2)
