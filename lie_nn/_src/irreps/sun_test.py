import numpy as np
import pytest
from lie_nn.irreps import SU2, SU2_
from lie_nn._src.irreps.sun import (
    Jz_matrices,
    S_to_Ms,
    clebsch_gordan_eldest,
    dim,
    index_to_M,
    lower_ladder_matrices,
    upper_ladder_matrices,
)
from lie_nn.util import commutator

j_max = 4


@pytest.mark.parametrize("j3", range(j_max + 1))
@pytest.mark.parametrize("j2", range(j_max + 1))
@pytest.mark.parametrize("j1", range(j_max + 1))
def test_highest_weight_constraint_with_su2(j1: int, j2: int, j3: int):
    S1 = SU2_((j1, 0))
    S2 = SU2_((j2, 0))
    S3 = SU2_((j3, 0))

    eldest_weight = list(S_to_Ms(S3.S))[-1]

    C1 = clebsch_gordan_eldest(S1.S, S2.S, eldest_weight)

    C2 = SU2.clebsch_gordan(SU2(j1), SU2(j2), SU2(j3))
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


@pytest.mark.parametrize("j3", range(j_max + 1))
@pytest.mark.parametrize("j2", range(j_max + 1))
@pytest.mark.parametrize("j1", range(j_max + 1))
def test_cg_su2(j1: int, j2: int, j3: int):
    S1 = SU2_((j1, 0))
    S2 = SU2_((j2, 0))
    S3 = SU2_((j3, 0))

    C1 = SU2_.clebsch_gordan(S1, S2, S3)
    C2 = SU2.clebsch_gordan(SU2(j1), SU2(j2), SU2(j3))

    np.testing.assert_allclose(C1, C2, rtol=1e-15, atol=1e-15)
