import numpy as np
import pytest
from lie_nn.irreps import SU2Rep, SURep
from lie_nn.irreps._sun import (
    Jz_matrices,
    S_to_Ms,
    construct_highest_weight_constraint,
    dim,
    index_to_M,
    lower_ladder_matrices,
    upper_ladder_matrices,
)
from lie_nn.util import commutator, null_space, round_to_sqrt_rational


j_max = 4


@pytest.mark.parametrize("j3", range(j_max + 1))
@pytest.mark.parametrize("j2", range(j_max + 1))
@pytest.mark.parametrize("j1", range(j_max + 1))
def test_highest_weight_constraint_with_su2(j1: int, j2: int, j3: int):
    S1 = SURep((j1, 0))
    S2 = SURep((j2, 0))
    S3 = SURep((j3, 0))

    eldest_weight = list(S_to_Ms(S3.S))[-1]

    A = construct_highest_weight_constraint(S1, S2, eldest_weight)
    C1 = null_space(A[:, ::-1], round_fn=round_to_sqrt_rational)[:, ::-1]  # [dim_null_space, dim_solution]
    C1 = C1.reshape(-1, S1.dim, S2.dim)

    C2 = SU2Rep.clebsch_gordan(SU2Rep(j1), SU2Rep(j2), SU2Rep(j3))
    C2 = C2[:, :, :, -1]  # eldest weight is the last index

    np.testing.assert_allclose(C1, C2)


Ss = [(0,)] + [(j, 0) for j in range(j_max + 1)] + [(1, 1, 0), (2, 1, 0), (2, 2, 0), (2, 2, 1, 0)]


@pytest.mark.parametrize("S", Ss)
def test_index(S):
    assert list(S_to_Ms(S)) == [index_to_M(S, i) for i in range(dim(S))]


@pytest.mark.parametrize("S", Ss)
def test_J_commutators(S):
    Jp = upper_ladder_matrices(S)
    Jm = lower_ladder_matrices(S)
    Jz = Jz_matrices(S)

    np.testing.assert_allclose(commutator(Jp, Jm), 2 * Jz)
    np.testing.assert_allclose(commutator(Jz, Jp), Jp)
    np.testing.assert_allclose(commutator(Jz, Jm), -Jm)
