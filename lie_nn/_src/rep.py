import dataclasses

import numpy as np
import scipy.linalg
from typing import Optional
from .util import infer_change_of_basis, kron, vmap, infer_algebra_from_generators, test_algebra_vs_generators


class Rep:
    r"""Abstract Class, Representation of a Lie group."""

    @property
    def lie_dim(self) -> int:
        A = self.algebra()
        d = A.shape[0]
        assert A.shape == (d, d, d)
        # X = self.continuous_generators()
        # assert X.shape[0] == d
        return d

    @property
    def dim(self) -> int:
        X = self.continuous_generators()
        d = X.shape[1]
        # H = self.discrete_generators()
        # assert X.shape[1:] == (d, d)
        # assert H.shape[1:] == (d, d)
        return d

    def algebra(self) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        raise NotImplementedError

    def continuous_generators(self) -> np.ndarray:
        raise NotImplementedError

    def discrete_generators(self) -> np.ndarray:
        raise NotImplementedError

    def create_trivial(self) -> "Rep":
        # Create a trivial representation from the same group as self
        raise NotImplementedError

    def exp_map(self, continuous_params: np.ndarray, discrete_params: np.ndarray) -> np.ndarray:
        """Instanciate the representation

        Args:
            continuous_params: ``(lie_dim,)`` array of continuous parameters
            discrete_params: ``(len(H),)`` array of discrete parameters (integers)

        Returns:
            ``(dim, dim)`` array
        """
        output = scipy.linalg.expm(np.einsum("a,aij->ij", continuous_params, self.continuous_generators()))
        for k, h in reversed(list(zip(discrete_params, self.discrete_generators()))):
            output = np.linalg.matrix_power(h, k) @ output
        return output

    def __repr__(self) -> str:
        return f"Rep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.discrete_generators())})"

    def is_trivial(self) -> bool:
        return self.dim == 1 and np.all(self.continuous_generators() == 0.0) and np.all(self.discrete_generators() == 1.0)

    @classmethod
    def clebsch_gordan(cls, rep1: "Rep", rep2: "Rep", rep3: "Rep", *, round_fn=lambda x: x) -> np.ndarray:
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

        X_in = vmap(lambda x1, x2: kron(x1.T, i2) + kron(i1, x2.T))(rep1.continuous_generators(), rep2.continuous_generators())
        X_out = vmap(lambda x3: x3.T)(rep3.continuous_generators())

        H_in = vmap(lambda x1, x2: kron(x1.T, x2.T), out_shape=(rep1.dim * rep2.dim, rep1.dim * rep2.dim))(
            rep1.discrete_generators(), rep2.discrete_generators()
        )
        H_out = vmap(lambda x3: x3.T, out_shape=(rep3.dim, rep3.dim))(rep3.discrete_generators())

        Y_in = np.concatenate([X_in, H_in])
        Y_out = np.concatenate([X_out, H_out])
        cg = infer_change_of_basis(Y_in, Y_out, round_fn=round_fn)
        np.testing.assert_allclose(
            np.einsum("aij,bjk->abik", Y_in, cg),
            np.einsum("bij,ajk->abik", cg, Y_out),
            rtol=1e-10,
            atol=1e-10,
        )

        assert cg.dtype in [np.float64, np.complex128], "Clebsch-Gordan coefficient must be computed with double precision."

        cg = round_fn(cg * np.sqrt(rep3.dim))
        cg = cg.reshape((-1, rep1.dim, rep2.dim, rep3.dim))
        return cg

    def test_algebra_vs_generators(rep: "Rep", rtol=1e-10, atol=1e-10):
        assert test_algebra_vs_generators(rep.algebra(), rep.continuous_generators(), rtol=rtol, atol=atol)

    @classmethod
    def test_clebsch_gordan_vs_generators(cls, rep1: "Rep", rep2: "Rep", rep3: "Rep", rtol=1e-10, atol=1e-10):
        X1 = rep1.continuous_generators()  # (lie_group_dimension, rep1.dim, rep1.dim)
        X2 = rep2.continuous_generators()  # (lie_group_dimension, rep2.dim, rep2.dim)
        X3 = rep3.continuous_generators()  # (lie_group_dimension, rep3.dim, rep3.dim)
        assert X1.shape[0] == X2.shape[0] == X3.shape[0]

        cg = cls.clebsch_gordan(rep1, rep2, rep3)
        assert cg.ndim == 1 + 3, (rep1, rep2, rep3, cg.shape)
        assert cg.shape == (cg.shape[0], rep1.dim, rep2.dim, rep3.dim)

        # Orthogonality
        # left_side = np.einsum('zijk,wijl->zkwl', cg, np.conj(cg))
        # right_side = np.eye(cg.shape[0] * rep3.dim).reshape((cg.shape[0], rep3.dim, cg.shape[0], rep3.dim))
        # np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)

        # if rep3 in rep1 * rep2:
        #     assert cg.shape[0] > 0
        # else:
        #     assert cg.shape[0] == 0

        left_side = np.einsum("zijk,dlk->zdijl", cg, X3)
        right_side = np.einsum("dil,zijk->zdljk", X1, cg) + np.einsum("djl,zijk->zdilk", X2, cg)

        for solution in range(cg.shape[0]):
            for i in range(X1.shape[0]):
                if not np.allclose(left_side[solution][i], right_side[solution][i], rtol=rtol, atol=atol):
                    print('Left side: einsum("zijk,dlk->zdijl", cg, X3)')
                    print(left_side[solution][i])
                    print('Right side: einsum("dil,zijk->zdljk", X1, cg) + einsum("djl,zijk->zdilk", X2, cg)')
                    print(right_side[solution][i])
                    raise AssertionError(
                        f"Solution {solution} of Clebsch-Gordan coefficient is not correct for Lie algebra generator {i}."
                    )


@dataclasses.dataclass
class GenericRep(Rep):
    r"""Unknown representation"""
    A: np.ndarray
    X: np.ndarray
    H: np.ndarray

    def from_rep(rep: Rep) -> "GenericRep":
        return GenericRep(rep.algebra(), rep.continuous_generators(), rep.discrete_generators())

    def from_generators(
        X: np.ndarray,
        H: Optional[np.ndarray] = None,
        round_fn=lambda x: x,
    ) -> Optional["GenericRep"]:
        A = infer_algebra_from_generators(X, round_fn=round_fn)
        if A is None:
            return None
        if H is None:
            H = np.zeros((0, X.shape[1], X.shape[1]))
        return GenericRep(A, X, H)

    def algebra(self) -> np.ndarray:
        """``[X_i, X_j] = A_ijk X_k``"""
        return self.A

    def continuous_generators(self) -> np.ndarray:
        return self.X

    def discrete_generators(self) -> np.ndarray:
        return self.H

    def create_trivial(self) -> "GenericRep":
        return GenericRep(
            A=self.algebra(),
            X=np.zeros((self.lie_dim, 1, 1)),
            H=np.ones((self.discrete_generators().shape[0], 1, 1)),
        )

    def __repr__(self) -> str:
        return f"GenericRep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.discrete_generators())})"
