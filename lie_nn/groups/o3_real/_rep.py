# book keeping for O3

import itertools
from dataclasses import dataclass
from typing import Iterator, List

import jax.numpy as jnp

from .. import Rep as RepBase
from ._cg import cg
from ._j import Jd


def _rot90_y(l):
    r"""90 degree rotation around Y axis."""
    M = jnp.zeros((2 * l + 1, 2 * l + 1))
    inds = jnp.arange(0, 2 * l + 1, 1)
    reversed_inds = jnp.arange(2 * l, -1, -1)
    frequencies = jnp.arange(l, -l - 1, -1)
    M = M.at[..., inds, reversed_inds].set(jnp.array([0, 1, 0, -1])[frequencies % 4])
    M = M.at[..., inds, inds].set(jnp.array([1, 0, -1, 0])[frequencies % 4])
    return M


@dataclass(frozen=True)
class Rep(RepBase):
    l: int
    p: int

    def __mul__(ir1: 'Rep', ir2: 'Rep') -> List['Rep']:
        # selection rule
        return [
            Rep(l, ir1.p * ir2.p)
            for l in range(abs(ir1.l - ir2.l), ir1.l + ir2.l + 1)
        ]

    @classmethod
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> jnp.ndarray:
        # return a numpy array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        if ir3 in ir1 * ir2:
            return cg[(ir1.l, ir2.l, ir3.l)][None]
        else:
            return jnp.zeros((0, ir1.dim, ir2.dim, ir3.dim))

    @property
    def dim(ir: 'Rep') -> int:
        return 2 * ir.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        # not sure if we need this
        for l in itertools.count():
            yield Rep(l, (-1)**l)
            yield Rep(l, -(-1)**l)

    def discrete_generators(ir: 'Rep') -> jnp.ndarray:
        return ir.p * jnp.eye(ir.dim)[None]

    def continuous_generators(ir: 'Rep') -> jnp.ndarray:
        inds = jnp.arange(0, ir.dim, 1)
        reversed_inds = jnp.arange(2 * ir.l, -1, -1)
        frequencies = jnp.arange(ir.l, -ir.l - 1, -1.0)
        K = _rot90_y(ir.l)

        Y = jnp.zeros((ir.dim, ir.dim)).at[..., inds, reversed_inds].set(frequencies)
        X = Jd[ir.l] @ Y @ Jd[ir.l]
        Z = K @ X @ K
        return jnp.stack([X, Y, Z], axis=0)
