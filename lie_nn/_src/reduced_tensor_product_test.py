from lie_nn import reduced_symmetric_tensor_product_basis
from lie_nn.irreps import SO3
from lie_nn import ReducedRep
import numpy as np


def test_tensor_product_basis_equivariance():
    irreps = ReducedRep.from_string("0+1+2+0", SO3)
    Q = reduced_symmetric_tensor_product_basis(irreps, 3)

    params = (0.2, 0.1, 0.13)

    D_out = Q.rep.exp_map(params, ())
    Q1 = np.einsum("ijkx,xy->ijky", Q.array, D_out)

    D_in = irreps.exp_map(params, ())
    Q2 = np.einsum("ijkx,li,mj,nk->lmnx", Q.array, D_in, D_in, D_in)

    np.testing.assert_allclose(Q1, Q2, atol=1e-6, rtol=1e-6)
