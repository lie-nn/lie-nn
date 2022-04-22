import numpy as np
from lie_nn.util import null_space


def test_null_space():
    A = np.random.normal(size=(130, 100)) + 1j * np.random.normal(size=(130, 100))
    B = np.random.normal(size=(100, 350)) + 1j * np.random.normal(size=(100, 350))
    S = A @ B
    X = null_space(S)

    assert np.allclose(S @ X.T, 0)
