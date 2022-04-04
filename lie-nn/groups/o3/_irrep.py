# book keeping for O3

import itertools
from dataclasses import dataclass
from typing import Iterator, List

import numpy as np


@dataclass(frozen=True)
class Irrep:
    l: int
    p: int

    # selection rule
    def __mul__(ir1: 'Irrep', ir2: 'Irrep') -> List['Irrep']:
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

    # not sure if we need this
    @classmethod
    def iterator(cls) -> Iterator['Irrep']:
        for l in itertools.count():
            yield Irrep(l, (-1)**l)
            yield Irrep(l, -(-1)**l)

    def discrete_generators(ir: 'Irrep') -> List[np.ndarray]:
        # return a list of one (2l+1) x (2l+1) matrix

        return [ir.p * np.eye(ir.dim)]

    def continuous_generators(ir: 'Irrep') -> List[np.ndarray]:
        # return a list of three (2l+1) x (2l+1) matrix
        # TODO: implement this
        # return [generator_x, generator_y, generator_z]
        pass
