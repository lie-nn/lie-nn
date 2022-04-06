import itertools

import pytest
from lie_nn.groups.o3_real import Rep as RepO3
from lie_nn.groups.so3_real import Rep as RepSO3
# from lie_nn.groups.so13 import Rep as RepSO13
from lie_nn.groups.su2 import Rep as RepSU2


@pytest.mark.parametrize('Rep', [RepO3, RepSU2, RepSO3])
def test_cg(Rep):
    irs = list(itertools.islice(Rep.iterator(), 6))

    Rep.test_clebsch_gordan(irs, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('Rep', [RepO3, RepSU2, RepSO3])
def test_algebra(Rep):
    irs = list(itertools.islice(Rep.iterator(), 6))

    for ir in irs:
        ir.test_algebra(atol=1e-3, rtol=1e-3)
