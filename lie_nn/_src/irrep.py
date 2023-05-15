from dataclasses import dataclass
from typing import Iterator


import numpy as np
from .rep import Rep


@dataclass(frozen=True)
class TabulatedIrrep(Rep):
    @classmethod
    def from_string(cls, string: str) -> "TabulatedIrrep":
        raise NotImplementedError

    def __mul__(rep1: "TabulatedIrrep", rep2: "TabulatedIrrep") -> Iterator["TabulatedIrrep"]:
        # Selection rule
        raise NotImplementedError

    @property
    def lie_dim(rep) -> int:
        return rep.algebra().shape[0]

    @property
    def dim(rep: "TabulatedIrrep") -> int:
        raise NotImplementedError

    def __lt__(rep1: "TabulatedIrrep", rep2: "TabulatedIrrep") -> bool:
        # This is used for sorting the irreps
        raise NotImplementedError

    @classmethod
    def iterator(cls) -> Iterator["TabulatedIrrep"]:
        # Requirements:
        #  - the first element must be the trivial representation
        #  - the elements must be sorted by the __lt__ method
        raise NotImplementedError

    @classmethod
    def create_trivial(cls) -> "TabulatedIrrep":
        return cls.iterator().__next__()

    def continuous_generators(rep: "TabulatedIrrep") -> np.ndarray:
        # return an array of shape ``(lie_group_dimension, rep.dim, rep.dim)``
        raise NotImplementedError

    def discrete_generators(rep: "TabulatedIrrep") -> np.ndarray:
        # return an array of shape ``(num_discrete_generators, rep.dim, rep.dim)``
        raise NotImplementedError

    def algebra(rep=None) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        pass

    @classmethod
    def clebsch_gordan(
        cls, rep1: "TabulatedIrrep", rep2: "TabulatedIrrep", rep3: "TabulatedIrrep"
    ) -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        raise NotImplementedError
