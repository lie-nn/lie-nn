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

    def __mul__(rep1: 'Rep', rep2: 'Rep') -> List['Rep']:
        return [
            Rep(l=l, p=rep1.p * rep2.p)
            for l in range(abs(rep1.l - rep2.l), rep1.l + rep2.l + 1)
            if l <= _lmax  # TODO: à discuter
        ]

    @classmethod
    def clebsch_gordan(cls, rep1: 'Rep', rep2: 'Rep', rep3: 'Rep') -> jnp.ndarray:
        # return an array of shape ``(dim_null_space, rep1.dim, rep2.dim, rep3.dim)``
        if rep3 in rep1 * rep2:
            if rep1.l <= rep2.l <= rep3.l:
                out = cg[(rep1.l, rep2.l, rep3.l)].reshape(rep1.dim, rep2.dim, rep3.dim)
            if rep1.l <= rep3.l <= rep2.l:
                out = cg[(rep1.l, rep3.l, rep2.l)].reshape(rep1.dim, rep3.dim, rep2.dim).transpose(0, 2, 1) * ((-1) ** (rep1.l + rep2.l + rep3.l))
            if rep2.l <= rep1.l <= rep3.l:
                out = cg[(rep2.l, rep1.l, rep3.l)].reshape(rep2.dim, rep1.dim, rep3.dim).transpose(1, 0, 2) * ((-1) ** (rep1.l + rep2.l + rep3.l))
            if rep3.l <= rep2.l <= rep1.l:
                out = cg[(rep3.l, rep2.l, rep1.l)].reshape(rep3.dim, rep2.dim, rep1.dim).transpose(2, 1, 0) * ((-1) ** (rep1.l + rep2.l + rep3.l))
            if rep2.l <= rep3.l <= rep1.l:
                out = cg[(rep2.l, rep3.l, rep1.l)].reshape(rep2.dim, rep3.dim, rep1.dim).transpose(2, 0, 1)
            if rep3.l <= rep1.l <= rep2.l:
                out = cg[(rep3.l, rep1.l, rep2.l)].reshape(rep3.dim, rep1.dim, rep2.dim).transpose(1, 2, 0)
            return out[None]
        else:
            return jnp.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    @property
    def dim(rep: 'Rep') -> int:
        return 2 * rep.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        for l in range(0, _lmax + 1):
            yield Rep(l=l, p=1)
            yield Rep(l=l, p=-1)

    def discrete_generators(rep: 'Rep') -> jnp.ndarray:
        return rep.p * jnp.eye(rep.dim)[None]

    def continuous_generators(rep: 'Rep') -> jnp.ndarray:
        inds = jnp.arange(0, rep.dim, 1)
        reversed_inds = jnp.arange(2 * rep.l, -1, -1)
        frequencies = jnp.arange(rep.l, -rep.l - 1, -1.0)
        K = _rot90_y(rep.l)

        Y = jnp.zeros((rep.dim, rep.dim)).at[..., inds, reversed_inds].set(frequencies)
        X = Jd[rep.l] @ Y @ Jd[rep.l]
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
