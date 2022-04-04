# book keeping for O3

import itertools
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np

from .. import Irrep as IrrepBase


@dataclass(frozen=True)
class Irrep(IrrepBase):
    l: int
    p: int

    def __mul__(ir1: 'Irrep', ir2: 'Irrep') -> List['Irrep']:
        # selection rule
        return [
            Irrep(l, ir1.p * ir2.p)
            for l in range(abs(ir1.l - ir2.l), ir1.l + ir2.l + 1)
        ]

    @classmethod
    def clebsch_gordan(cls, ir1: 'Irrep', ir2: 'Irrep', ir3: 'Irrep') -> np.ndarray:
        # return a numpy array of shape ``(dim_null_space, ir1.dim, ir2.dim, ir3.dim)``
        if ir3 in ir1 * ir2:
            pass
            # return ...
        else:
            return np.zeros((0, ir1.dim, ir2.dim, ir3.dim))

    @property
    def dim(ir: 'Irrep') -> int:
        return 2 * ir.l + 1

    @classmethod
    def iterator(cls) -> Iterator['Irrep']:
        # not sure if we need this
        for l in itertools.count():
            yield Irrep(l, (-1)**l)
            yield Irrep(l, -(-1)**l)

    def discrete_generators(ir: 'Irrep') -> np.ndarray:
        return ir.p * np.eye(ir.dim)[None]

    def continuous_generators(ir: 'Irrep') -> np.ndarray:
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
