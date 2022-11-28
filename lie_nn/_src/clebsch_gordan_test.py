import numpy as np

import lie_nn as lie


def test_cg_irrep():
    Q = np.array([[0, 1, 1], [2, 2, 1], [-3, 1, 0]])
    rep1 = lie.change_basis(lie.irreps.SU2(2), Q)

    Q = np.array([[1, 2, 0], [2, 1, 0], [0, -1, 1.0]])
    rep2 = lie.change_basis(lie.irreps.SU2(2), Q)

    Q = np.array([[1, 0, 0], [1, 1, -1], [0, 0, 1.0]])
    rep3 = lie.change_basis(lie.irreps.SU2(2), Q)

    lie.check_representation_triplet(rep1, rep2, rep3)


def test_cg_generic():
    Q = np.array([[0, 1, 1], [2, 2, 1], [-3, 1, 0]])
    rep1 = lie.change_basis(lie.irreps.SO3(1), Q)

    Q = np.random.randn(5, 5)
    rep2 = lie.change_basis(lie.irreps.SO3(2), Q)

    Q = np.array([[1, 0, 0], [1, 1, -1], [0, 0, 1.0]])
    rep3 = lie.change_basis(lie.irreps.SO3(1), Q)

    lie.check_representation_triplet(rep1, rep2, rep3)
