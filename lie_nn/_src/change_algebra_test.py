import numpy as np

import lie_nn as lie


def test_change_algebra():
    rep = lie.irreps.SU2(2)

    Q = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]) / np.sqrt(2.0)
    rep = lie.change_algebra(rep, Q)

    rep.test_algebra_vs_generators()
    lie.clebsch_gordan_vs_generators_test(rep, rep, rep)
