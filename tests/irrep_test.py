import itertools

import numpy as np
import pytest
from lie_nn import Irrep
from lie_nn.irreps import O3Rep, SL2Rep, SO3Rep, SO13Rep, SU2RealRep, SU2Rep, SU2Rep_, SU3Rep, SU4Rep
from lie_nn.util import round_to_sqrt_rational


REPRESENTATIONS = [O3Rep, SU2Rep, SO3Rep, SU2RealRep, SL2Rep, SO13Rep, SU2Rep_, SU3Rep, SU4Rep]


def bunch_of_reps():
    return sum((list(itertools.islice(IR.iterator(), 7)) for IR in REPRESENTATIONS), [])


def bunch_of_triplets():
    return sum((list(itertools.product(itertools.islice(IR.iterator(), 5), repeat=3)) for IR in REPRESENTATIONS), [])


@pytest.mark.parametrize("ir", bunch_of_reps())
def test_algebra_vs_generators(ir: Irrep):
    ir.test_algebra_vs_generators()


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_clebsch_gordan_vs_generators(ir1: Irrep, ir2: Irrep, ir3: Irrep):
    ir1.test_clebsch_gordan_vs_generators(ir1, ir2, ir3)


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_recompute_clebsch_gordan(ir1: Irrep, ir2: Irrep, ir3: Irrep):
    tol = 1e-14
    C1 = Irrep.clebsch_gordan(ir1, ir2, ir3, round_fn=round_to_sqrt_rational)
    C2 = ir1.clebsch_gordan(ir1, ir2, ir3)
    assert np.allclose(C1, C2, atol=tol, rtol=tol) or np.allclose(C1, -C2, atol=tol, rtol=tol)


@pytest.mark.parametrize("ir1, ir2, ir3", bunch_of_triplets())
def test_selection_rule(ir1: Irrep, ir2: Irrep, ir3: Irrep):
    cg = ir1.clebsch_gordan(ir1, ir2, ir3)

    if ir3 in ir1 * ir2:
        assert cg.shape[0] > 0
    else:
        assert cg.shape[0] == 0
