# book keeping for O3

import itertools
from dataclasses import dataclass
from typing import List

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

    @property
    def dim(ir) -> int:
        return 2 * ir.l + 1

    # not sure if we need this
    @classmethod
    def iterator(cls):
        r"""Iterator through all the irreps of :math:`O(3)`

        Examples:
            >>> it = Irrep.iterator()
            >>> next(it), next(it), next(it), next(it)
            (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1)**l)
            yield Irrep(l, -(-1)**l)

    def discrete_generators(ir):
        # return a list of one (2l+1) x (2l+1) matrix

        return [ir.p * np.eye(ir.dim)]

    def continuous_generators(ir):
        # return a list of three (2l+1) x (2l+1) matrix
        # TODO: implement this
        # return [generator_x, generator_y, generator_z]
        pass
