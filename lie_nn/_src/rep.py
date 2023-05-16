import dataclasses
from typing import Optional

import numpy as np
import scipy.linalg

from .util import infer_algebra_from_generators, check_algebra_vs_generators


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

    @property
    def A(self) -> np.ndarray:
        return self.algebra()

    def continuous_generators(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def X(self) -> np.ndarray:
        return self.continuous_generators()

    def discrete_generators(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def H(self) -> np.ndarray:
        return self.discrete_generators()

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
        output = scipy.linalg.expm(
            np.einsum("a,aij->ij", continuous_params, self.continuous_generators())
        )
        for k, h in reversed(list(zip(discrete_params, self.discrete_generators()))):
            output = np.linalg.matrix_power(h, k) @ output
        return output

    def __repr__(self) -> str:
        return f"Rep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.H)})"

    def is_trivial(self) -> bool:
        return (
            self.dim == 1
            and np.all(self.continuous_generators() == 0.0)
            and np.all(self.discrete_generators() == 1.0)
        )

    def check_algebra_vs_generators(rep: "Rep", rtol=1e-10, atol=1e-10):
        check_algebra_vs_generators(
            rep.algebra(), rep.continuous_generators(), rtol=rtol, atol=atol, assert_=True
        )


@dataclasses.dataclass(init=False)
class GenericRep(Rep):
    r"""Unknown representation"""
    _A: np.ndarray
    _X: np.ndarray
    _H: np.ndarray

    def __init__(self, A: np.ndarray, X: np.ndarray, H: np.ndarray):
        self._A = A
        self._X = X
        self._H = H

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
        return self._A

    def continuous_generators(self) -> np.ndarray:
        return self._X

    def discrete_generators(self) -> np.ndarray:
        return self._H

    def create_trivial(self) -> "GenericRep":
        return GenericRep(
            A=self.algebra(),
            X=np.zeros((self.lie_dim, 1, 1)),
            H=np.ones((self.discrete_generators().shape[0], 1, 1)),
        )

    def __repr__(self) -> str:
        return f"GenericRep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.H)})"


def check_representation_triplet(rep1: Rep, rep2: Rep, rep3: Rep, rtol=1e-10, atol=1e-10):
    assert np.allclose(rep1.algebra(), rep2.algebra(), rtol=rtol, atol=atol)
    assert np.allclose(rep1.algebra(), rep3.algebra(), rtol=rtol, atol=atol)

    rep1.check_algebra_vs_generators(rtol=rtol, atol=atol)
    rep2.check_algebra_vs_generators(rtol=rtol, atol=atol)
    rep3.check_algebra_vs_generators(rtol=rtol, atol=atol)

    X1 = rep1.continuous_generators()  # (lie_group_dimension, rep1.dim, rep1.dim)
    X2 = rep2.continuous_generators()  # (lie_group_dimension, rep2.dim, rep2.dim)
    X3 = rep3.continuous_generators()  # (lie_group_dimension, rep3.dim, rep3.dim)
    assert X1.shape[0] == X2.shape[0] == X3.shape[0]

    from .clebsch_gordan import clebsch_gordan

    cg = clebsch_gordan(rep1, rep2, rep3)
    assert cg.ndim == 1 + 3, (rep1, rep2, rep3, cg.shape)
    assert cg.shape == (cg.shape[0], rep1.dim, rep2.dim, rep3.dim)

    # Orthogonality
    # left_side = np.einsum('zijk,wijl->zkwl', cg, np.conj(cg))
    # right_side = np.eye(cg.shape[0] * rep3.dim)
    # .reshape((cg.shape[0], rep3.dim, cg.shape[0], rep3.dim))
    # np.testing.assert_allclose(left_side, right_side, rtol=rtol, atol=atol)

    # if rep3 in rep1 * rep2:
    #     assert cg.shape[0] > 0
    # else:
    #     assert cg.shape[0] == 0

    left_side = np.einsum("zijk,dlk->zdijl", cg, X3)
    right_side = np.einsum("dil,zijk->zdljk", X1, cg) + np.einsum("djl,zijk->zdilk", X2, cg)

    for solution in range(cg.shape[0]):
        for i in range(X1.shape[0]):
            if not np.allclose(
                left_side[solution][i], right_side[solution][i], rtol=rtol, atol=atol
            ):
                np.set_printoptions(precision=3, suppress=True)
                print(rep1, rep2, rep3)
                print('Left side: einsum("zijk,dlk->zdijl", cg, X3)')
                print(left_side[solution][i])
                print(
                    "Right side: "
                    'einsum("dil,zijk->zdljk", X1, cg) + einsum("djl,zijk->zdilk", X2, cg)'
                )
                print(right_side[solution][i])
                np.set_printoptions(precision=8, suppress=False)
                raise AssertionError(
                    f"Solution {solution}/{cg.shape[0]} for {rep1} * {rep2} = {rep3} "
                    "is not correct."
                    f"Clebsch-Gordan coefficient is not correct for Lie algebra generator {i}."
                )
