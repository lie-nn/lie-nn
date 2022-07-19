import itertools

import numpy as np
import pytest
from lie_nn import Irrep, O3Rep, SL2Rep, SO3Rep, SO13Rep, SU2Rep, SU2RealRep
from lie_nn.util import round_to_sqrt_rational


def first_reps(Rep: Irrep, n: int):
    return list(itertools.islice(Rep.iterator(), n))


REPRESENTATIONS = [O3Rep, SU2Rep, SO3Rep, SU2RealRep, SL2Rep, SO13Rep]


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_cg_equivariance(Rep: Irrep):
    reps = first_reps(Rep, 4)

    Rep.test_clebsch_gordan(reps)


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_recompute_cg(Rep: Irrep):
    reps = first_reps(Rep, 4)
    tol = 1e-14

    for rep1, rep2, rep3 in itertools.product(reps, reps, reps):
        C1 = Irrep.clebsch_gordan(rep1, rep2, rep3, round_fn=round_to_sqrt_rational)
        C2 = Rep.clebsch_gordan(rep1, rep2, rep3)
        assert np.allclose(C1, C2, atol=tol, rtol=tol) or np.allclose(C1, -C2, atol=tol, rtol=tol)


@pytest.mark.parametrize("rep", sum((first_reps(Rep, 6) for Rep in REPRESENTATIONS), []))
def test_algebra(rep: Irrep):
    rep.test_algebra()


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_selection_rule(Rep: Irrep):
    reps = list(itertools.islice(Rep.iterator(), 6))

    for rep1, rep2, rep3 in itertools.product(reps, repeat=3):
        cg = Rep.clebsch_gordan(rep1, rep2, rep3)

        if rep3 in rep1 * rep2:
            assert cg.shape[0] > 0
        else:
            assert cg.shape[0] == 0
