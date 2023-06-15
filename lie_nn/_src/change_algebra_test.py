import numpy as np

import lie_nn as lie


def test_change_algebra():
    rep = lie.irreps.SU2(2)

    Q = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]) / np.sqrt(2.0)
    rep = lie.change_algebra(rep, Q)

    lie.test.check_algebra_vs_generators(rep.A, rep.X)
    lie.test.check_representation_triplet(rep, rep, rep)
