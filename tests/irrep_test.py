import itertools

import numpy as np
import pytest
from lie_nn import Irrep, O3Rep, SL2Rep, SO3Rep, SO13Rep, SU2Rep, SU2RealRep
from lie_nn.util import round_to_sqrt_rational


def first_reps(IR: Irrep, n: int):
    return list(itertools.islice(IR.iterator(), n))


REPRESENTATIONS = [O3Rep, SU2Rep, SO3Rep, SU2RealRep, SL2Rep, SO13Rep]


@pytest.mark.parametrize("IR", REPRESENTATIONS)
def test_cg_equivariance(IR: Irrep):
    reps = first_reps(IR, 4)

    IR.test_clebsch_gordan(reps)


@pytest.mark.parametrize("IR", REPRESENTATIONS)
def test_recompute_cg(IR: Irrep):
    reps = first_reps(IR, 4)
    tol = 1e-14

    for rep1, rep2, rep3 in itertools.product(reps, reps, reps):
        C1 = Irrep.clebsch_gordan(rep1, rep2, rep3, round_fn=round_to_sqrt_rational)
        C2 = IR.clebsch_gordan(rep1, rep2, rep3)
        assert np.allclose(C1, C2, atol=tol, rtol=tol) or np.allclose(C1, -C2, atol=tol, rtol=tol)


@pytest.mark.parametrize("ir", sum((first_reps(IR, 6) for IR in REPRESENTATIONS), []))
def test_algebra(ir: Irrep):
    ir.test_algebra()


@pytest.mark.parametrize("IR", REPRESENTATIONS)
def test_selection_rule(IR: Irrep):
    reps = list(itertools.islice(IR.iterator(), 6))

    for ir1, ir2, ir3 in itertools.product(reps, repeat=3):
        cg = IR.clebsch_gordan(ir1, ir2, ir3)

        if ir3 in ir1 * ir2:
            assert cg.shape[0] > 0
        else:
            assert cg.shape[0] == 0
