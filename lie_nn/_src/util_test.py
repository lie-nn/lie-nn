import numpy as np
from lie_nn.util import (
    nullspace,
    infer_change_of_basis,
    round_to_sqrt_rational,
)

from lie_nn._src.util import (
    as_approx_integer_ratio,
    limit_denominator,
    normalize_integer_ratio,
)
from fractions import Fraction
from functools import partial


def test_null_space():
    A = np.random.normal(size=(130, 100)) + 1j * np.random.normal(size=(130, 100))
    B = np.random.normal(size=(100, 350)) + 1j * np.random.normal(size=(100, 350))
    S = A @ B
    X = nullspace(S)

    assert np.allclose(S @ X.T, 0)


def test_infer_change_of_basis():
    n, d = 5, 10
    X2 = np.random.normal(size=(n, d, d)) + 1j * np.random.normal(size=(n, d, d))
    S = np.random.normal(size=(d, d)) + 1j * np.random.normal(size=(d, d))
    X1 = S @ X2 @ np.linalg.inv(S)

    T = infer_change_of_basis(X1, X2)
    assert T.shape == (1, d, d)
    T = T[0]

    assert np.allclose(X1, T @ X2 @ np.linalg.inv(T))

    # S and T are eqaul up to a global phase
    np.testing.assert_allclose(
        S / S[0, 0],
        T / T[0, 0],
    )


def test_as_integer_ratio():
    x = np.linspace(-100, 100, 5435)
    n, d = as_approx_integer_ratio(x)
    assert np.abs(n / d - x).max() < 1e-13
    assert np.all(np.gcd(n, d) == 1)

    x = np.array([0, 1, 1 / 2, -1 / 4, 1 / 2 + 1 / 4 - 1 / 8])
    n, d = as_approx_integer_ratio(x)
    assert np.abs(n / d - x).max() < 1e-13
    assert np.all(np.gcd(n, d) == 1)


def test_limit_denominator():
    @partial(np.vectorize)
    def _limit_denominator(n, d) -> float:
        x = Fraction(n, d).limit_denominator()
        return x.numerator, x.denominator

    n = np.random.randint(-1_000_000, 1_000_000, size=(200_000,))
    d = np.random.randint(1, 1_000_000, size=n.shape)
    n = np.concatenate([n, np.array([-906339, -386764])])
    d = np.concatenate([d, np.array([8850, 1509])])

    n1, d1 = normalize_integer_ratio(n, d)
    n1, d1 = limit_denominator(n1, d1)
    n2, d2 = _limit_denominator(n, d)
    assert np.all(n1 == n2)
    assert np.all(d1 == d2)


def test_round_to_sqrt_rational():
    @partial(np.vectorize, otypes=[np.float64])
    def _round_to_sqrt_rational(x: float) -> float:
        sign = 1 if x >= 0 else -1
        return sign * Fraction(x**2).limit_denominator() ** 0.5

    n = np.random.randint(-1_000_000, 1_000_000, size=(200_000,))
    d = np.random.randint(1, 1_000_000, size=n.shape)
    n = np.concatenate([n, np.array([-906339, -386764])])
    d = np.concatenate([d, np.array([8850, 1509])])

    x = np.sign(n) * np.sqrt(np.abs(n) / d)

    y = round_to_sqrt_rational(x)
    y_ = _round_to_sqrt_rational(x)
    np.testing.assert_allclose(y, y_, rtol=1e-15, atol=1e-15)
