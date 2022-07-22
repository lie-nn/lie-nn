from dataclasses import dataclass
from typing import Iterator, List


import numpy as np
from . import Rep
from .util import commutator, kron, vmap, change_of_basis


@dataclass(frozen=True)
class Irrep(Rep):
    def __mul__(rep1: "Irrep", rep2: "Irrep") -> Iterator["Irrep"]:
        # Selection rule
        raise NotImplementedError

    @classmethod
    def clebsch_gordan(cls, rep1: "Irrep", rep2: "Irrep", rep3: "Irrep", *, round_fn=lambda x: x) -> np.ndarray:
        r"""Computes the Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).

        Args:
            rep1: The first input representation.
            rep2: The second input representation.
            rep3: The output representation.

        Returns:
            The Clebsch-Gordan coefficient of the triplet (rep1, rep2, rep3).
            It is an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``.
        """
        # Check the group structure
        assert np.allclose(rep1.algebra(), rep2.algebra())
        assert np.allclose(rep2.algebra(), rep3.algebra())

        i1 = np.eye(rep1.dim)
        i2 = np.eye(rep2.dim)

        X_in = vmap(lambda x1, x2: kron(x1, i2) + kron(i1, x2))(rep1.continuous_generators(), rep2.continuous_generators())
        X_out = rep3.continuous_generators()

        H_in = vmap(lambda x1, x2: kron(x1, x2), out_shape=(rep1.dim * rep2.dim, rep1.dim * rep2.dim))(
            rep1.discrete_generators(), rep2.discrete_generators()
        )
        H_out = rep3.discrete_generators()

        cg = change_of_basis(np.concatenate([X_in, H_in]), np.concatenate([X_out, H_out]), round_fn=round_fn)

        assert cg.dtype in [np.float64, np.complex128], "Clebsch-Gordan coefficient must be computed with double precision."

        cg = cg * np.sqrt(rep3.dim)
        cg = cg.reshape((-1, rep1.dim, rep2.dim, rep3.dim))
        return cg

    @property
    def lie_dim(rep) -> int:
        return rep.algebra().shape[0]

    @property
    def dim(rep: "Irrep") -> int:
        raise NotImplementedError

    @classmethod
    def iterator(cls) -> Iterator["Irrep"]:
        # not sure if we need this
        raise NotImplementedError

    def continuous_generators(rep: "Irrep") -> np.ndarray:
        # return an array of shape ``(lie_group_dimension, rep.dim, rep.dim)``
        raise NotImplementedError

    def discrete_generators(rep: "Irrep") -> np.ndarray:
        # return an array of shape ``(num_discrete_generators, rep.dim, rep.dim)``
        raise NotImplementedError

    def algebra(rep) -> np.ndarray:
        # [X_i, X_j] = A_ijk X_k
        pass

    def test_algebra(rep: "Irrep", rtol=1e-10, atol=1e-10):
        X = rep.continuous_generators()  # (lie_group_dimension, rep.dim, rep.dim)
        left_side = vmap(vmap(commutator, (0, None), 0), (None, 0), 1)(X, X)
        right_side = np.einsum("ijk,kab->ijab", rep.algebra(), X)
        assert np.allclose(left_side, right_side, rtol=rtol, atol=atol)

    @classmethod
    def test_clebsch_gordan(cls, reps: List["Irrep"], rtol=1e-10, atol=1e-10):
        for rep1 in reps:
            for rep2 in reps:
                for rep3 in reps:
                    X1 = rep1.continuous_generators()  # (lie_group_dimension, rep1.dim, rep1.dim)
                    X2 = rep2.continuous_generators()  # (lie_group_dimension, rep2.dim, rep2.dim)
                    X3 = rep3.continuous_generators()  # (lie_group_dimension, rep3.dim, rep3.dim)

                    cg = cls.clebsch_gordan(rep1, rep2, rep3)
                    assert cg.ndim == 1 + 3, (rep1, rep2, rep3, cg.shape)
                    assert cg.shape == (cg.shape[0], rep1.dim, rep2.dim, rep3.dim)

                    # Orthogonality
                    # left_side = np.einsum('zijk,wijl->zkwl', cg, np.conj(cg))
                    # right_side = np.eye(cg.shape[0] * rep3.dim).reshape((cg.shape[0], rep3.dim, cg.shape[0], rep3.dim))
                    # assert np.allclose(left_side, right_side, rtol=rtol, atol=atol)

                    if rep3 in rep1 * rep2:
                        assert cg.shape[0] > 0
                    else:
                        assert cg.shape[0] == 0

                    left_side = np.einsum("zijk,dlk->zdijl", cg, X3)
                    right_side = np.einsum("dil,zijk->zdljk", X1, cg) + np.einsum("djl,zijk->zdilk", X2, cg)
                    assert np.allclose(left_side, right_side, rtol=rtol, atol=atol)
