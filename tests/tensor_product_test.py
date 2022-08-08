import itertools

import numpy as np
import pytest
from lie_nn import GenericRep, Irrep, MulIrrep, ReducedRep, Rep, tensor_product
from lie_nn.irreps import O3Rep, SL2Rep, SO3Rep, SO13Rep, SU2Rep


def first_reps(IR: Irrep, n: int):
    return list(itertools.islice(IR.iterator(), n))


REPRESENTATIONS = [O3Rep, SU2Rep, SO3Rep, SL2Rep, SO13Rep]
# TODO: add SU2RealRep and SUNRep
# Note: SU2RealRep are not Irreps and this might be a problem


def cast_to_generic(rep: Rep) -> GenericRep:
    return GenericRep(
        A=rep.algebra(),
        X=rep.continuous_generators(),
        H=rep.discrete_generators(),
    )


@pytest.mark.parametrize("ir1, ir2", sum((list(itertools.product(first_reps(IR, 4), repeat=2)) for IR in REPRESENTATIONS), []))
def test_tensor_product_consistency(ir1, ir2):
    rep1 = ReducedRep.from_irreps([(2, ir1), ir2])
    rep2 = ReducedRep.from_irreps([(3, ir1)])

    tp1 = tensor_product(rep1, rep2)
    tp2 = tensor_product(cast_to_generic(rep1), cast_to_generic(rep2))

    np.testing.assert_allclose(
        tp1.continuous_generators(),
        tp2.continuous_generators(),
        atol=1e-10,
    )


def test_tensor_product_types():
    assert isinstance(tensor_product(O3Rep(l=1, p=1), O3Rep(l=1, p=1)), ReducedRep)
    assert isinstance(tensor_product(O3Rep(l=1, p=1), MulIrrep(mul=2, rep=O3Rep(l=1, p=1))), ReducedRep)
