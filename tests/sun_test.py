import numpy as np
import pytest
from lie_nn.irreps import SU2Rep, SURep
from lie_nn.irreps._sun import S_to_Ms, construct_highest_weight_constraint
from lie_nn.util import null_space, round_to_sqrt_rational

j_max = 5


@pytest.mark.parametrize("j3", range(j_max + 1))
@pytest.mark.parametrize("j2", range(j_max + 1))
@pytest.mark.parametrize("j1", range(j_max + 1))
def test_highest_weight_constraint_with_su2(j1: int, j2: int, j3: int):
    S1 = SURep((j1, 0))
    S2 = SURep((j2, 0))
    S3 = SURep((j3, 0))

    eldest_weight = next(S_to_Ms(S3.S))

    A = construct_highest_weight_constraint(S1, S2, eldest_weight)
    C1 = null_space(A, round_fn=round_to_sqrt_rational)  # [dim_null_space, dim_solution]
    C1 = C1.reshape(-1, S1.dim, S2.dim)

    C2 = SU2Rep.clebsch_gordan(SU2Rep(j1), SU2Rep(j2), SU2Rep(j3))
    C2 = C2[:, ::-1, ::-1, -1]  # eldest weight is the last index

    np.testing.assert_allclose(C1, C2)
