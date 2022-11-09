from dataclasses import dataclass
from typing import Iterator


import numpy as np
from .rep import Rep
from .util import commutator, vmap


@dataclass(frozen=True)
class Irrep(Rep):
    @classmethod
    def from_string(cls, string: str) -> "Irrep":
        raise NotImplementedError

    def __mul__(rep1: "Irrep", rep2: "Irrep") -> Iterator["Irrep"]:
        # Selection rule
        raise NotImplementedError

    @property
    def lie_dim(rep) -> int:
        return rep.algebra().shape[0]

    @property
    def dim(rep: "Irrep") -> int:
        raise NotImplementedError

    def __lt__(rep1: "Irrep", rep2: "Irrep") -> bool:
        # This is used for sorting the irreps
        raise NotImplementedError

    @classmethod
    def iterator(cls) -> Iterator["Irrep"]:
        # Requirements:
        #  - the first element must be the trivial representation
        #  - the elements must be sorted by the __lt__ method
        raise NotImplementedError

    @classmethod
    def create_trivial(cls) -> "Irrep":
        return cls.iterator().__next__()

    def continuous_generators(rep: "Irrep") -> np.ndarray:
        # return an array of shape ``(lie_group_dimension, rep.dim, rep.dim)``
        raise NotImplementedError

    def discrete_generators(rep: "Irrep") -> np.ndarray:
        # return an array of shape ``(num_discrete_generators, rep.dim, rep.dim)``
        raise NotImplementedError

    def algebra(rep=None) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        pass

    def test_algebra_vs_generators(rep: "Irrep", rtol=1e-10, atol=1e-10):
        X = rep.continuous_generators()  # (lie_group_dimension, rep.dim, rep.dim)
        left_side = vmap(vmap(commutator, (0, None), 0), (None, 0), 1)(X, X)
        right_side = np.einsum("ijk,kab->ijab", rep.algebra(), X)
        np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)

    @classmethod
    def test_clebsch_gordan_vs_generators(cls, rep1: "Irrep", rep2: "Irrep", rep3: "Irrep", rtol=1e-10, atol=1e-10):
        X1 = rep1.continuous_generators()  # (lie_group_dimension, rep1.dim, rep1.dim)
        X2 = rep2.continuous_generators()  # (lie_group_dimension, rep2.dim, rep2.dim)
        X3 = rep3.continuous_generators()  # (lie_group_dimension, rep3.dim, rep3.dim)

        cg = cls.clebsch_gordan(rep1, rep2, rep3)
        assert cg.ndim == 1 + 3, (rep1, rep2, rep3, cg.shape)
        assert cg.shape == (cg.shape[0], rep1.dim, rep2.dim, rep3.dim)

        # Orthogonality
        # left_side = np.einsum('zijk,wijl->zkwl', cg, np.conj(cg))
        # right_side = np.eye(cg.shape[0] * rep3.dim).reshape((cg.shape[0], rep3.dim, cg.shape[0], rep3.dim))
        # np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)

        if rep3 in rep1 * rep2:
            assert cg.shape[0] > 0
        else:
            assert cg.shape[0] == 0

        left_side = np.einsum("zijk,dlk->zdijl", cg, X3)
        right_side = np.einsum("dil,zijk->zdljk", X1, cg) + np.einsum("djl,zijk->zdilk", X2, cg)
        np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)
