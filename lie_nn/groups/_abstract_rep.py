from dataclasses import dataclass
from typing import Iterator, List

import jax
import jax.numpy as jnp
import numpy as np
from lie_nn.util import commutator, kron, vmap, change_of_basis


def static_jax_pytree(cls):
    cls = dataclass(frozen=True)(cls)
    jax.tree_util.register_pytree_node(cls, lambda x: ((), x), lambda x, _: x)
    return cls


@jax.jit
def matrix_power(F, n):
    upper_limit = 32
    init_carry = n, F, jnp.eye(F.shape[0])

    def body(carry, _):
        # One step of the iteration
        n, z, result = carry
        new_n, bit = jnp.divmod(n, 2)

        new_result = jax.lax.cond(bit, lambda x: z @ x, lambda x: x, result)

        # No more computation necessary if n = 0
        # Is there a better way to early break rather than just returning something empty?
        new_z = jax.lax.cond(new_n, lambda z: z @ z, lambda _: jnp.empty(z.shape), z)

        return (new_n, new_z, new_result), None

    result = jax.lax.cond(n == 1, lambda _: F, lambda _: jax.lax.scan(body, init_carry, None, length=upper_limit)[0][2], None)

    return result


@static_jax_pytree
class AbstractRep:
    def __mul__(rep1: "AbstractRep", rep2: "AbstractRep") -> List["AbstractRep"]:
        # selection rule
        pass

    @classmethod
    def clebsch_gordan(cls, rep1: "AbstractRep", rep2: "AbstractRep", rep3: "AbstractRep") -> np.ndarray:
        r"""Computes the Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).

        Args:
            rep1: The first input representation.
            rep2: The second input representation.
            rep3: The output representation.

        Returns:
            The Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).
            It is an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``.
        """
        i1 = np.eye(rep1.dim)
        i2 = np.eye(rep2.dim)

        X_in = vmap(lambda x1, x2: kron(x1, i2) + kron(i1, x2))(rep1.continuous_generators(), rep2.continuous_generators())
        X_out = rep3.continuous_generators()
        cg = change_of_basis(X_in, X_out)

        assert cg.dtype in [np.float64, np.complex128], "Clebsch-Gordan coefficient must be computed with double precision."

        cg = cg * np.sqrt(rep3.dim)
        cg = cg.reshape((-1, rep1.dim, rep2.dim, rep3.dim))
        return cg

    @property
    def dim(rep: "AbstractRep") -> int:
        pass

    @classmethod
    def iterator(cls) -> Iterator["AbstractRep"]:
        # not sure if we need this
        pass

    def continuous_generators(rep: "AbstractRep") -> np.ndarray:
        # return an array of shape ``(lie_group_dimension, rep.dim, rep.dim)``
        pass

    def discrete_generators(rep: "AbstractRep") -> np.ndarray:
        # return an array of shape ``(num_discrete_generators, rep.dim, rep.dim)``
        pass

    @classmethod
    def algebra(cls) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        pass

    def exp_map(rep: "AbstractRep", continuous_params: jnp.ndarray, discrete_params: jnp.ndarray) -> jnp.ndarray:
        # return a matrix of shape ``(rep.dim, rep.dim)``
        discrete = jax.vmap(matrix_power)(rep.discrete_generators(), discrete_params)
        output = jax.scipy.linalg.expm(jnp.einsum("a,aij->ij", continuous_params, rep.continuous_generators()))
        for x in reversed(discrete):
            output = x @ output
        return output

    def test_algebra(rep: "AbstractRep", rtol=1e-05, atol=1e-08):
        X = rep.continuous_generators()  # (lie_group_dimension, rep.dim, rep.dim)
        left_side = vmap(vmap(commutator, (0, None), 0), (None, 0), 1)(X, X)
        right_side = np.einsum("ijk,kab->ijab", rep.algebra(), X)
        assert np.allclose(left_side, right_side, rtol=rtol, atol=atol)

    @classmethod
    def test_clebsch_gordan(cls, reps: List["AbstractRep"], rtol=1e-05, atol=1e-08):
        for rep1 in reps:
            for rep2 in reps:
                for rep3 in reps:
                    X1 = rep1.continuous_generators()  # (lie_group_dimension, rep1.dim, rep1.dim)
                    X2 = rep2.continuous_generators()  # (lie_group_dimension, rep2.dim, rep2.dim)
                    X3 = rep3.continuous_generators()  # (lie_group_dimension, rep3.dim, rep3.dim)

                    cg = cls.clebsch_gordan(rep1, rep2, rep3)
                    assert cg.ndim == 1 + 3, (rep1, rep2, rep3, cg.shape)
                    assert cg.shape == (cg.shape[0], rep1.dim, rep2.dim, rep3.dim)

                    # Orthogonality
                    # left_side = np.einsum('zijk,wijl->zkwl', cg, np.conj(cg))
                    # right_side = np.eye(cg.shape[0] * rep3.dim).reshape((cg.shape[0], rep3.dim, cg.shape[0], rep3.dim))
                    # assert np.allclose(left_side, right_side, rtol=rtol, atol=atol)

                    if rep3 in rep1 * rep2:
                        assert cg.shape[0] > 0
                    else:
                        assert cg.shape[0] == 0

                    left_side = np.einsum("zijk,dlk->zdijl", cg, X3)
                    right_side = np.einsum("dil,zijk->zdljk", X1, cg) + np.einsum("djl,zijk->zdilk", X2, cg)
                    assert np.allclose(left_side, right_side, rtol=rtol, atol=atol)
