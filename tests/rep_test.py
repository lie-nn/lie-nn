import itertools

import pytest
from lie_nn.groups import O3Rep, SO3Rep, SU2Rep, SL2Rep, SO13Rep

REPRESENTATIONS = [O3Rep, SU2Rep, SO3Rep]  # TODO add SL2Rep and SO13Rep


@pytest.mark.parametrize('Rep', REPRESENTATIONS)
def test_cg(Rep):
    reps = list(itertools.islice(Rep.iterator(), 4))

    Rep.test_clebsch_gordan(reps, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('Rep', REPRESENTATIONS)
def test_algebra(Rep):
    reps = list(itertools.islice(Rep.iterator(), 6))

    for rep in reps:
        rep.test_algebra(atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('Rep', REPRESENTATIONS + [SL2Rep, SO13Rep])
def test_selection_rule(Rep):
    reps = list(itertools.islice(Rep.iterator(), 6))

    for rep1, rep2, rep3 in itertools.product(reps, repeat=3):
        cg = Rep.clebsch_gordan(rep1, rep2, rep3)

        if rep3 in rep1 * rep2:
            assert cg.shape[0] > 0
        else:
            assert cg.shape[0] == 0
