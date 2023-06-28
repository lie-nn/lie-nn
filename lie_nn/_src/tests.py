import numpy as np
from .rep import Rep
from .utils import check_algebra_vs_generators


def check_representation_triplet(rep1: Rep, rep2: Rep, rep3: Rep, rtol=1e-10, atol=1e-10):
    assert np.allclose(rep1.algebra(), rep2.algebra(), rtol=rtol, atol=atol)
    assert np.allclose(rep1.algebra(), rep3.algebra(), rtol=rtol, atol=atol)

    check_algebra_vs_generators(rep1.A, rep1.X, rtol=rtol, atol=atol)
    check_algebra_vs_generators(rep2.A, rep2.X, rtol=rtol, atol=atol)
    check_algebra_vs_generators(rep3.A, rep3.X, rtol=rtol, atol=atol)

    X1 = rep1.continuous_generators()  # (lie_group_dimension, rep1.dim, rep1.dim)
    X2 = rep2.continuous_generators()  # (lie_group_dimension, rep2.dim, rep2.dim)
    X3 = rep3.continuous_generators()  # (lie_group_dimension, rep3.dim, rep3.dim)
    assert X1.shape[0] == X2.shape[0] == X3.shape[0]

    from .clebsch_gordan import clebsch_gordan

    cg = clebsch_gordan(rep1, rep2, rep3)
    assert cg.ndim == 1 + 3, (rep1, rep2, rep3, cg.shape)
    assert cg.shape == (cg.shape[0], rep1.dim, rep2.dim, rep3.dim)

    # Orthogonality
    # left_side = np.einsum('zijk,wijl->zkwl', cg, np.conj(cg))
    # right_side = np.eye(cg.shape[0] * rep3.dim)
    # .reshape((cg.shape[0], rep3.dim, cg.shape[0], rep3.dim))
    # np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)

    # if rep3 in rep1 * rep2:
    #     assert cg.shape[0] > 0
    # else:
    #     assert cg.shape[0] == 0

    left_side = np.einsum("zijk,dlk->zdijl", cg, X3)
    right_side = np.einsum("dil,zijk->zdljk", X1, cg) + np.einsum("djl,zijk->zdilk", X2, cg)

    for solution in range(cg.shape[0]):
        for i in range(X1.shape[0]):
            if not np.allclose(
                left_side[solution][i], right_side[solution][i], rtol=rtol, atol=atol
            ):
                np.set_printoptions(precision=3, suppress=True)
                print(rep1, rep2, rep3)
                print('Left side: einsum("zijk,dlk->zdijl", cg, X3)')
                print(left_side[solution][i])
                print(
                    "Right side: "
                    'einsum("dil,zijk->zdljk", X1, cg) + einsum("djl,zijk->zdilk", X2, cg)'
                )
                print(right_side[solution][i])
                diff = left_side[solution][i] - right_side[solution][i]
                print("Difference:", np.abs(diff).max())
                np.set_printoptions(precision=8, suppress=False)
                raise AssertionError(
                    f"Solution {solution}/{cg.shape[0]} for {rep1} * {rep2} = {rep3} "
                    "is not correct."
                    f"Clebsch-Gordan coefficient is not correct for Lie algebra generator {i}."
                )
