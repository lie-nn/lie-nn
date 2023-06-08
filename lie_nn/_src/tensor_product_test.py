import itertools

import numpy as np
import pytest
from lie_nn import GenericRep, TabulatedIrrep, MulRep, SumRep, tensor_product, tensor_power
from lie_nn.irreps import O3, SL2C, SO3, SO13, SU2


def first_reps(IR: TabulatedIrrep, n: int):
    return list(itertools.islice(IR.iterator(), n))


REPRESENTATIONS = [O3, SU2, SO3, SL2C, SO13]
# TODO: add SU2Real (or remove it completely)
# Note: SU2Real are not Irreps and this might be a problem
# TODO: resolve tensor_product_consistency for SU3, SU4


@pytest.mark.parametrize(
    "ir1, ir2",
    sum((list(itertools.product(first_reps(IR, 4), repeat=2)) for IR in REPRESENTATIONS), []),
)
def test_tensor_product_consistency(ir1, ir2):
    rep1 = SumRep([MulRep(2, ir1), ir2])
    rep2 = SumRep([MulRep(3, ir1)])

    tp1 = tensor_product(rep1, rep2)
    tp2 = tensor_product(GenericRep.from_rep(rep1), GenericRep.from_rep(rep2))

    np.testing.assert_allclose(tp1.X, tp2.X, atol=1e-10)


def test_tensor_product_types():
    assert isinstance(tensor_product(O3(l=1, p=1), O3(l=1, p=1)), SumRep)
    assert isinstance(tensor_product(O3(l=1, p=1), MulRep(mul=2, rep=O3(l=1, p=1))), SumRep)


def test_tensor_power_types():
    assert isinstance(tensor_power(O3(l=1, p=1), 2), SumRep)
    assert isinstance(tensor_power(MulRep(mul=2, rep=O3(l=1, p=1)), 2), SumRep)
    assert isinstance(tensor_power(GenericRep.from_rep(O3(l=1, p=1)), 2), GenericRep)
