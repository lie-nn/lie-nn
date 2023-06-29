import lie_nn as lie


def test_fourvector():
    vec = lie.irreps.SO13.four_vector()

    lie.test.check_representation_triplet(vec, vec, vec)
