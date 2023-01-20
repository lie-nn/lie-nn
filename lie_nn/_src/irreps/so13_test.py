import lie_nn as lie


def test_fourvector():
    vec = lie.irreps.SO13.four_vector()

    vec.check_algebra_vs_generators()
    lie.check_representation_triplet(vec, vec, vec)
