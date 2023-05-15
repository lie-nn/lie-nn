import itertools

import numpy as np
import pytest
from lie_nn import TabulatedIrrep, clebsch_gordan, check_representation_triplet, GenericRep
from lie_nn.irreps import O3, SL2C, SO3, SO13, SU2Real, SU2, SU2_, SU3, SU4
from lie_nn.util import round_to_sqrt_rational


REPRESENTATIONS = [O3, SU2, SO3, SU2Real, SL2C, SO13, SU2_, SU3, SU4]


def bunch_of_reps():
    return sum((list(itertools.islice(IR.iterator(), 7)) for IR in REPRESENTATIONS), [])


def bunch_of_triplets():
    return sum(
        (
            list(itertools.product(itertools.islice(IR.iterator(), 4), repeat=3))
            for IR in REPRESENTATIONS
        ),
        [],
    )


@pytest.mark.parametrize("ir", bunch_of_reps())
def test_algebra_vs_generators(ir: TabulatedIrrep):
    ir.check_algebra_vs_generators()


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_numerical_cg_vs_generators(ir1: TabulatedIrrep, ir2: TabulatedIrrep, ir3: TabulatedIrrep):
    check_representation_triplet(GenericRep.from_rep(ir1), ir2, ir3)


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_irreps_clebsch_gordan_vs_generators(
    ir1: TabulatedIrrep, ir2: TabulatedIrrep, ir3: TabulatedIrrep
):
    check_representation_triplet(ir1, ir2, ir3)


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_recompute_clebsch_gordan(ir1: TabulatedIrrep, ir2: TabulatedIrrep, ir3: TabulatedIrrep):
    tol = 1e-14
    C1 = clebsch_gordan(ir1, ir2, ir3, round_fn=round_to_sqrt_rational)
    C2 = ir1.clebsch_gordan(ir1, ir2, ir3)
    assert np.allclose(C1, C2, atol=tol, rtol=tol) or np.allclose(C1, -C2, atol=tol, rtol=tol)


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_selection_rule(ir1: TabulatedIrrep, ir2: TabulatedIrrep, ir3: TabulatedIrrep):
    cg = ir1.clebsch_gordan(ir1, ir2, ir3)

    if ir3 in ir1 * ir2:
        assert cg.shape[0] > 0
    else:
        assert cg.shape[0] == 0


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_trivial(Rep):
    assert Rep.create_trivial().is_trivial()

    ir = Rep.create_trivial()
    assert ir.create_trivial().is_trivial()


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_order(Rep):
    irreps = list(itertools.islice(Rep.iterator(), 10))
    for ir1, ir2 in zip(irreps, irreps[1:]):
        assert ir1 < ir2
