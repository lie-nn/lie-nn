import numpy as np

import lie_nn as lie


def is_unitary(rep: lie.Rep) -> bool:
    X = rep.continuous_generators()
    H = rep.discrete_generators()
    H_unit = np.allclose(H @ np.conj(np.transpose(H, (0, 2, 1))), np.eye(rep.dim), atol=1e-13)
    # exp(X) @ exp(X^H) = 1
    # X + X^H = 0
    X_unit = np.allclose(X + np.conj(np.transpose(X, (0, 2, 1))), 0, atol=1e-13)
    return H_unit and X_unit
