import lie_nn as lie


def test_group_product_SO3_Z2_Z2():
    r1oo = lie.group_product(lie.irreps.SO3(l=1), lie.irreps.Z2(p=-1), lie.irreps.Z2(p=-1))
    r2eo = lie.group_product(lie.irreps.SO3(l=2), lie.irreps.Z2(p=1), lie.irreps.Z2(p=-1))
    r2ee = lie.group_product(lie.irreps.SO3(l=2), lie.irreps.Z2(p=1), lie.irreps.Z2(p=1))

    lie.test.check_representation_triplet(r1oo, r1oo, r2ee)
    lie.test.check_representation_triplet(r1oo, r2eo, r2ee)


def test_group_product_SU2_SU2():
    j1 = lie.irreps.SU2(j=1)  # spin 1/2
    j2 = lie.irreps.SU2(j=2)  # spin 1
    j3 = lie.irreps.SU2(j=3)  # spin 3/2

    j11 = lie.group_product(j1, j1)
    j12 = lie.group_product(j1, j2)
    j13 = lie.group_product(j1, j3)
    j22 = lie.group_product(j2, j2)

    lie.test.check_representation_triplet(j11, j11, j22)
    lie.test.check_representation_triplet(j11, j12, j22)
    lie.test.check_representation_triplet(j12, j13, j22)
