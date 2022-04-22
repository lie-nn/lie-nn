import itertools

import numpy as np
import pytest
from lie_nn.groups import AbstractRep, O3Rep, SL2Rep, SO3Rep, SO13Rep, SU2Rep, SU2RealRep


def first_reps(Rep, n):
    return list(itertools.islice(Rep.iterator(), n))


REPRESENTATIONS = [O3Rep, SU2Rep, SO3Rep, SU2RealRep, SL2Rep, SO13Rep]


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_cg_equivariance(Rep):
    reps = first_reps(Rep, 4)

    Rep.test_clebsch_gordan(reps, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_recompute_cg(Rep):
    reps = first_reps(Rep, 4)

    for rep1, rep2, rep3 in itertools.product(reps, reps, reps):
        C1 = AbstractRep.clebsch_gordan(rep1, rep2, rep3)
        C2 = Rep.clebsch_gordan(rep1, rep2, rep3)
        assert np.allclose(C1, C2, atol=1e-3, rtol=1e-3) or np.allclose(C1, -C2, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("rep", sum((first_reps(Rep, 6) for Rep in REPRESENTATIONS), []))
def test_algebra(rep):
    rep.test_algebra(atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("Rep", REPRESENTATIONS)
def test_selection_rule(Rep):
    reps = list(itertools.islice(Rep.iterator(), 6))

    for rep1, rep2, rep3 in itertools.product(reps, repeat=3):
        cg = Rep.clebsch_gordan(rep1, rep2, rep3)

        if rep3 in rep1 * rep2:
            assert cg.shape[0] > 0
        else:
            assert cg.shape[0] == 0
