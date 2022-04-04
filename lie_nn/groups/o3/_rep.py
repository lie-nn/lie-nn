# book keeping for O3

import itertools
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np

from .. import Rep as RepBase


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
    def clebsch_gordan(cls, ir1: 'Rep', ir2: 'Rep', ir3: 'Rep') -> np.ndarray:
        # return a numpy array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        if ir3 in ir1 * ir2:
            pass
            # return ...
        else:
            return np.zeros((0, ir1.dim, ir2.dim, ir3.dim))

    @property
    def dim(ir: 'Rep') -> int:
        return 2 * ir.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Rep']:
        # not sure if we need this
        for l in itertools.count():
            yield Rep(l, (-1)**l)
            yield Rep(l, -(-1)**l)

    def discrete_generators(ir: 'Rep') -> np.ndarray:
        return ir.p * np.eye(ir.dim)[None]

    def continuous_generators(ir: 'Rep') -> np.ndarray:
        if ir.l == 0:
            return np.ones((3, 1, 1))
        if ir.l == 1:
            return np.array([
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ],
                [
                    [0, 0, -1],
                    [0, 0, 0],
                    [1, 0, 0]
                ],
                [
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 0.0]
                ],
            ])
        # TODO: implement the rest
