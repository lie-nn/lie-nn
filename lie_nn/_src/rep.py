from typing import Iterator, Optional, Tuple, Type

import numpy as np
import scipy.linalg

from .util import check_algebra_vs_generators, direct_sum, infer_algebra_from_generators


class Rep:
    r"""Abstract Class, Representation of a Lie group."""

    def algebra(self) -> np.ndarray:
        """Array of shape [lie_dim, lie_dim, lie_dim]

        Satisfying the Lie algebra commutation relations:

        .. math::

            [X_i, X_j] = A_{ijk} X_k

        """
        raise NotImplementedError

    def continuous_generators(self) -> np.ndarray:
        """Array of shape [lie_dim, dim, dim]"""
        raise NotImplementedError

    def discrete_generators(self) -> np.ndarray:
        """Array of shape [len(H), dim, dim]"""
        raise NotImplementedError

    def create_trivial(self) -> "Rep":
        """Create trivial representation (dim=1)

        With the same algebra and len(H)
        """
        raise NotImplementedError

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

    @property
    def A(self) -> np.ndarray:
        return self.algebra()

    @property
    def X(self) -> np.ndarray:
        return self.continuous_generators()

    @property
    def H(self) -> np.ndarray:
        return self.discrete_generators()

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


class Irrep(Rep):
    pass


class TabulatedIrrep(Rep):
    @classmethod
    def from_string(cls, string: str) -> "TabulatedIrrep":
        raise NotImplementedError

    def __mul__(rep1: "TabulatedIrrep", rep2: "TabulatedIrrep") -> Iterator["TabulatedIrrep"]:
        # Selection rule
        raise NotImplementedError

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

    @classmethod
    def clebsch_gordan(
        cls, rep1: "TabulatedIrrep", rep2: "TabulatedIrrep", rep3: "TabulatedIrrep"
    ) -> np.ndarray:
        # return an array of shape ``(number_of_paths, rep1.dim, rep2.dim, rep3.dim)``
        raise NotImplementedError


class MulRep(Rep):
    mul: int
    rep: Rep

    def __init__(self, mul: int, rep: Rep):
        self.mul = mul
        self.rep = rep

    @classmethod
    def from_string(cls, string: str, cls_irrep: Type[Rep]) -> "MulRep":
        if "x" in string:
            mul, rep = string.split("x")
        else:
            mul, rep = 1, string
        return cls(mul=int(mul), rep=cls_irrep.from_string(rep))

    @property
    def dim(self) -> int:
        return self.mul * self.rep.dim

    def algebra(self) -> np.ndarray:
        return self.rep.algebra()

    def continuous_generators(self) -> np.ndarray:
        X = self.rep.X
        if X.shape[0] == 0:
            return np.empty((0, self.dim, self.dim))
        return np.stack([direct_sum(*[x for _ in range(self.mul)]) for x in X], axis=0)

    def discrete_generators(self) -> np.ndarray:
        H = self.rep.H
        if H.shape[0] == 0:
            return np.empty((0, self.dim, self.dim))
        return np.stack([direct_sum(*[x for _ in range(self.mul)]) for x in H], axis=0)

    def create_trivial(self) -> Rep:
        return self.rep.create_trivial()

    def __repr__(self) -> str:
        return f"{self.mul}x{self.rep}"


class SumRep(Rep):
    r"""Representation of the form

    .. math::
        \osum_i \rho_i
    """
    reps: Tuple[Rep, ...]

    def __init__(self, reps: Tuple[Rep, ...]):
        assert len(reps) >= 1
        self.reps = tuple(reps)

    @classmethod
    def from_string(cls, string: str, cls_irrep: Type[TabulatedIrrep]) -> "SumRep":
        return cls([MulRep.from_string(term, cls_irrep) for term in string.split("+")])

    @property
    def dim(self) -> int:
        return sum(rep.dim for rep in self.reps)

    def algebra(self) -> np.ndarray:
        return self.reps[0].algebra()

    def continuous_generators(self) -> np.ndarray:
        if self.lie_dim == 0:
            return np.empty((0, self.dim, self.dim))
        Xs = []
        for i in range(self.lie_dim):
            Xs += [direct_sum(*[rep.X[i] for rep in self.reps])]
        return np.stack(Xs)

    def discrete_generators(self) -> np.ndarray:
        n = len(self.reps[0].H)
        if n == 0:
            return np.empty((0, self.dim, self.dim))
        Hs = []
        for i in range(n):
            Hs += [direct_sum(*[rep.H[i] for rep in self.reps])]
        return np.stack(Hs)

    def create_trivial(self) -> Rep:
        return self.reps[0].create_trivial()

    def __repr__(self) -> str:
        r = " + ".join(repr(rep) for rep in self.reps)
        return r


class QRep(Rep):
    r"""Representation of the form

    .. math::

        Q \rho Q^{-1}
    """
    rep: Rep
    Q: np.ndarray

    def __init__(self, Q: np.ndarray, rep: Rep):
        self.rep = rep
        self.Q = Q

    @property
    def dim(self) -> int:
        return self.rep.dim

    def algebra(self) -> np.ndarray:
        return self.rep.algebra()

    def continuous_generators(self) -> np.ndarray:
        return np.einsum("ij,ajk,kl->ail", self.Q, self.rep.X, np.linalg.pinv(self.Q))

    def discrete_generators(self) -> np.ndarray:
        return np.einsum("ij,ajk,kl->ail", self.Q, self.rep.H, np.linalg.pinv(self.Q))

    def create_trivial(self) -> "Rep":
        return self.rep.create_trivial()

    def __repr__(self) -> str:
        return f"Q({self.rep})Q^{{-1}}"


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
