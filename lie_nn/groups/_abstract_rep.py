from typing import Iterator, List

import chex
import jax
import jax.numpy as jnp


@jax.jit
def matrix_power(F, n, upper_limit=32):
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

    result = jax.lax.cond(
        n == 1,
        lambda _: F,
        lambda _: jax.lax.scan(body, init_carry, None, length=upper_limit)[0][2],
        None
    )

    return result


def commutator(A, B):
    return A @ B - B @ A


@chex.dataclass(frozen=True)
class AbstractRep:
    def __mul__(ir1: 'AbstractRep', ir2: 'AbstractRep') -> List['AbstractRep']:
        # selection rule
        pass

    @classmethod
    def clebsch_gordan(cls, ir1: 'AbstractRep', ir2: 'AbstractRep', ir3: 'AbstractRep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        pass

    @property
    def dim(ir: 'AbstractRep') -> int:
        pass

    @classmethod
    def iterator(cls) -> Iterator['AbstractRep']:
        # not sure if we need this
        pass

    def continuous_generators(ir: 'AbstractRep') -> jnp.ndarray:
        # return an array of shape ``(lie_group_dimension, ir.dim, ir.dim)``
        pass

    def discrete_generators(ir: 'AbstractRep') -> jnp.ndarray:
        # return an array of shape ``(num_discrete_generators, ir.dim, ir.dim)``
        pass

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        pass

    def exp_map(ir: 'AbstractRep', continuous_params: jnp.ndarray, discrete_params: jnp.ndarray) -> jnp.ndarray:
        # return a matrix of shape ``(ir.dim, ir.dim)``
        discrete = jax.vmap(matrix_power)(ir.discrete_generators(), discrete_params)
        output = jax.scipy.linalg.expm(jnp.einsum('a,aij->ij', continuous_params, ir.continuous_generators()))
        for x in reversed(discrete):
            output = x @ output
        return output

    def test_algebra(ir: 'AbstractRep', rtol=1e-05, atol=1e-08) -> jnp.ndarray:
        X = ir.continuous_generators()  # (lie_group_dimension, ir.dim, ir.dim)
        left_side = jax.vmap(jax.vmap(commutator, (0, None), 0), (None, 0), 1)(X, X)
        right_side = jnp.einsum('ijk,kab->ijab', ir.algebra(), X)
        return jnp.allclose(left_side, right_side, rtol=rtol, atol=atol)

    @classmethod
    def test_clebsch_gordan(cls, irs: List['AbstractRep'], rtol=1e-05, atol=1e-08):
        for ir1 in irs:
            for ir2 in irs:
                for ir3 in irs:
                    X1 = ir1.continuous_generators()  # (lie_group_dimension, ir1.dim, ir1.dim)
                    X2 = ir2.continuous_generators()  # (lie_group_dimension, ir2.dim, ir2.dim)
                    X3 = ir3.continuous_generators()  # (lie_group_dimension, ir3.dim, ir3.dim)

                    print(ir1, ir2, ir3)
                    cg = cls.clebsch_gordan(ir1, ir2, ir3)
                    assert cg.ndim == 1 + 3, (ir1, ir2, ir3, cg.shape)
                    assert cg.shape == (cg.shape[0], ir1.dim, ir2.dim, ir3.dim)
                    if ir3 not in ir1 * ir2:
                        assert cg.shape[0] == 0

                    left_side = jnp.einsum('zijk,dlk->zdijl', cg, X3)
                    right_side = jnp.einsum('dil,zijk->zdljk', X1, cg) + jnp.einsum('djl,zijk->zdilk', X2, cg)
                    assert jnp.allclose(left_side, right_side, rtol=rtol, atol=atol)
