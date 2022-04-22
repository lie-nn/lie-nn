import numpy as np
from lie_nn.util import null_space, change_of_basis


def test_null_space():
    A = np.random.normal(size=(130, 100)) + 1j * np.random.normal(size=(130, 100))
    B = np.random.normal(size=(100, 350)) + 1j * np.random.normal(size=(100, 350))
    S = A @ B
    X = null_space(S)

    assert np.allclose(S @ X.T, 0)


def test_change_of_basis():
    n, d = 5, 10
    X2 = np.random.normal(size=(n, d, d)) + 1j * np.random.normal(size=(n, d, d))
    S = np.random.normal(size=(d, d)) + 1j * np.random.normal(size=(d, d))
    S = S / np.linalg.norm(S)
    X1 = S @ X2 @ np.linalg.inv(S)

    T = change_of_basis(X1, X2)
    assert np.allclose(X1, T @ X2 @ np.linalg.inv(T))
