from typing import Iterator, List

import chex
import jax.numpy as jnp

from .. import AbstractRep
from ._cg import cg
from ._j import Jd

_lmax = 8


def _rot90_y(l):
    r"""90 degree rotation around Y axis."""
    M = jnp.zeros((2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1)
    M = M.at[..., inds, reversed_inds].set(jnp.array([0, 1, 0, -1])[frequencies % 4])
    M = M.at[..., inds, inds].set(jnp.array([1, 0, -1, 0])[frequencies % 4])
    return M


@chex.dataclass(frozen=True)
class Rep(AbstractRep):
    l: int
    p: int

    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        return [
            Rep(l=l, p=ir1.p * ir2.p)
            for l in range(abs(ir1.l - ir2.l), ir1.l + ir2.l + 1)
            if l <= _lmax  # TODO: à discuter
        ]

    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        if ir3 in ir1 * ir2:
            if ir1.l <= ir2.l <= ir3.l:
                out = cg[(ir1.l, ir2.l, ir3.l)].reshape(ir1.dim, ir2.dim, ir3.dim)
            if ir1.l <= ir3.l <= ir2.l:
                out = cg[(ir1.l, ir3.l, ir2.l)].reshape(ir1.dim, ir3.dim, ir2.dim).transpose(0, 2, 1) * ((-1) ** (ir1.l + ir2.l + ir3.l))
            if ir2.l <= ir1.l <= ir3.l:
                out = cg[(ir2.l, ir1.l, ir3.l)].reshape(ir2.dim, ir1.dim, ir3.dim).transpose(1, 0, 2) * ((-1) ** (ir1.l + ir2.l + ir3.l))
            if ir3.l <= ir2.l <= ir1.l:
                out = cg[(ir3.l, ir2.l, ir1.l)].reshape(ir3.dim, ir2.dim, ir1.dim).transpose(2, 1, 0) * ((-1) ** (ir1.l + ir2.l + ir3.l))
            if ir2.l <= ir3.l <= ir1.l:
                out = cg[(ir2.l, ir3.l, ir1.l)].reshape(ir2.dim, ir3.dim, ir1.dim).transpose(2, 0, 1)
            if ir3.l <= ir1.l <= ir2.l:
                out = cg[(ir3.l, ir1.l, ir2.l)].reshape(ir3.dim, ir1.dim, ir2.dim).transpose(1, 2, 0)
            return out[None]
        else:
            return jnp.zeros((0, ir1.dim, ir2.dim, ir3.dim))

    @property
    def dim(ir: 'Rep') -> int:
        return 2 * ir.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        for l in range(0, _lmax + 1):
            yield Rep(l=l, p=1)
            yield Rep(l=l, p=-1)

    def discrete_generators(ir: 'Rep') -> jnp.ndarray:
        return ir.p * jnp.eye(ir.dim)[None]

    def continuous_generators(ir: 'Rep') -> jnp.ndarray:
        inds = jnp.arange(0, ir.dim, 1)
        reversed_inds = jnp.arange(2 * ir.l, -1, -1)
        frequencies = jnp.arange(ir.l, -ir.l - 1, -1.0)
        K = _rot90_y(ir.l)

        Y = jnp.zeros((ir.dim, ir.dim)).at[..., inds, reversed_inds].set(frequencies)
        X = Jd[ir.l] @ Y @ Jd[ir.l]
        Z = K.T @ X @ K
        return jnp.stack([X, Y, Z], axis=0)

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
