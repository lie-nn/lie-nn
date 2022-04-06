import itertools
from typing import Iterator, List

import chex
import jax.numpy as jnp

from . import AbstractRep
from .su2 import Rep as SU2Rep


def change_basis_r2c(l):
    q = jnp.zeros((2 * l + 1, 2 * l + 1), dtype=jnp.complex64)
    for m in range(-l, 0):
        w = -1j * 1j**m / jnp.sqrt(2)
        q = q.at[l + m, l + m].set(w)
        q = q.at[l - m, l + m].set(-w)

    for m in range(1, l + 1):
        w = 1j**(-m) / jnp.sqrt(2)
        q = q.at[l + m, l + m].set(w)
        q = q.at[l - m, l + m].set(w)

    q = q.at[l, l].set(1)
    return q


@chex.dataclass(frozen=True)
class Rep(AbstractRep):
    l: int

    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        return [Rep(l=l) for l in range(abs(ir1.l - ir2.l), ir1.l + ir2.l + 1, 1)]

    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        C = SU2Rep.clebsch_gordan(SU2Rep(j=ir1.l), SU2Rep(j=ir2.l), SU2Rep(j=ir3.l))
        Q1 = change_basis_r2c(ir1.l)
        Q2 = change_basis_r2c(ir2.l)
        Q3 = change_basis_r2c(ir3.l)
        C = jnp.einsum('ij,kl,mn,zikn->zjlm', Q1, Q2, jnp.conj(Q3.T), C)

        # make it real
        C = 1j**(ir1.l + ir2.l + ir3.l) * C
        assert jnp.all(jnp.abs(jnp.imag(C)) < 1e-5)
        C = jnp.real(C)

        # normalization
        C = C / jnp.linalg.norm(C)
        return C

    @property
    def dim(ir: 'Rep') -> int:
        return 2 * ir.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        for l in itertools.count(0):
            yield Rep(l=l)

    def continuous_generators(ir: 'Rep') -> jnp.ndarray:
        X = SU2Rep(j=ir.l).continuous_generators()
        Q = change_basis_r2c(ir.l)
        X = jnp.conj(Q.T) @ X @ Q
        assert jnp.max(jnp.abs(jnp.imag(X))) < 1e-5
        return jnp.real(X)

    def discrete_generators(ir: 'Rep') -> jnp.ndarray:
        return jnp.zeros((0, ir.dim, ir.dim))

    @classmethod
    def algebra(cls) -> jnp.ndarray:
        # [X_i, X_j] = A_ijk X_k
        return jnp.array([
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            [
                [0, 0, -1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 0.0],
            ],
        ])
