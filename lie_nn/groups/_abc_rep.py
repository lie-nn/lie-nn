# import abc
from dataclasses import dataclass
from typing import Iterator, List

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


@dataclass(frozen=True)
class Rep:  # (abc.ABC):
    # @abc.abstractmethod
    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        # selection rule
        pass

    # @abc.abstractmethod
    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return a numpy array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        pass

    # @abc.abstractmethod
    @property
    def dim(ir: 'Rep') -> int:
        pass

    # @abc.abstractmethod
    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        # not sure if we need this
        pass

    # @abc.abstractmethod
    def discrete_generators(ir: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(lie_group_dimension, ir.dim, ir.dim)``
        pass

    # @abc.abstractmethod
    def continuous_generators(ir: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(num_discrete_generators, ir.dim, ir.dim)``
        pass

    def exp_map(ir: 'Rep', continuous_params: jnp.ndarray, discrete_params: jnp.ndarray) -> jnp.ndarray:
        # return a matrix of shape ``(ir.dim, ir.dim)``
        discrete = jax.vmap(matrix_power)(ir.discrete_generators(), discrete_params)
        output = jax.scipy.linalg.expm(jnp.einsum('a,aij->ij', continuous_params, ir.continuous_generators()))
        for x in reversed(discrete):
            output = x @ output
        return output
