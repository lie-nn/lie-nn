import itertools

import numpy as np
import pytest
from lie_nn.irreps import O3, SL2C, SO3, SO13, SU2
import lie_nn as lie


def first_reps(IR: lie.TabulatedIrrep, n: int):
    return list(itertools.islice(IR.iterator(), n))


REPRESENTATIONS = [O3, SU2, SO3, SL2C, SO13]
# TODO: add SU2Real (or remove it completely)
# Note: SU2Real are not Irreps and this might be a problem
# TODO: resolve tensor_product_consistency for SU3, SU4


@pytest.mark.parametrize(
    "ir1, ir2",
    sum((list(itertools.product(first_reps(IR, 3), repeat=2)) for IR in REPRESENTATIONS), []),
)
def test_tensor_product_consistency(ir1, ir2):
    rep1 = lie.direct_sum(lie.multiply(2, ir1), ir2)
    rep2 = lie.direct_sum(lie.multiply(3, ir1))

    tp1 = lie.tensor_product(rep1, rep2)
    tp2 = lie.tensor_product(lie.GenericRep.from_rep(rep1), lie.GenericRep.from_rep(rep2))

    np.testing.assert_allclose(tp1.X, tp2.X, atol=1e-10)


def test_tensor_product_types():
    assert isinstance(lie.tensor_product(O3(l=1, p=1), O3(l=1, p=1)), lie.QRep)
    assert isinstance(lie.tensor_product(O3(l=1, p=1), lie.multiply(2, O3(l=1, p=1))), lie.QRep)


def test_tensor_power_types():
    assert isinstance(lie.tensor_power(O3(l=1, p=1), 2), lie.QRep)
    assert isinstance(lie.tensor_power(lie.multiply(2, O3(l=1, p=1)), 2), lie.QRep)
    assert isinstance(lie.tensor_power(lie.GenericRep.from_rep(O3(l=1, p=1)), 2), lie.GenericRep)
