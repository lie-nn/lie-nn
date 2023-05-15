from dataclasses import dataclass
import jax
import jax.numpy as jnp
from .irrep import TabulatedIrrep


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

    result = jax.lax.cond(
        n == 1,
        lambda _: F,
        lambda _: jax.lax.scan(body, init_carry, None, length=upper_limit)[0][2],
        None,
    )

    return result


def exp_map(
    rep: "TabulatedIrrep", continuous_params: jnp.ndarray, discrete_params: jnp.ndarray
) -> jnp.ndarray:
    # return a matrix of shape ``(rep.dim, rep.dim)``
    discrete = jax.vmap(matrix_power)(rep.discrete_generators(), discrete_params)
    output = jax.scipy.linalg.expm(
        jnp.einsum("a,aij->ij", continuous_params, rep.continuous_generators())
    )
    for x in reversed(discrete):
        output = x @ output
    return output
