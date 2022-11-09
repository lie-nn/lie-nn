import dataclasses

import numpy as np
import scipy.linalg

from .util import infer_change_of_basis, kron, vmap


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

        X_in = vmap(lambda x1, x2: kron(x1, i2) + kron(i1, x2))(rep1.continuous_generators(), rep2.continuous_generators())
        X_out = rep3.continuous_generators()

        H_in = vmap(lambda x1, x2: kron(x1, x2), out_shape=(rep1.dim * rep2.dim, rep1.dim * rep2.dim))(
            rep1.discrete_generators(), rep2.discrete_generators()
        )
        H_out = rep3.discrete_generators()

        cg = infer_change_of_basis(np.concatenate([X_in, H_in]), np.concatenate([X_out, H_out]), round_fn=round_fn)

        assert cg.dtype in [np.float64, np.complex128], "Clebsch-Gordan coefficient must be computed with double precision."

        cg = round_fn(cg * np.sqrt(rep3.dim))
        cg = cg.reshape((-1, rep1.dim, rep2.dim, rep3.dim))
        return cg


@dataclasses.dataclass
class GenericRep(Rep):
    r"""Unknown representation"""
    A: np.ndarray
    X: np.ndarray
    H: np.ndarray

    def from_rep(rep: Rep) -> "GenericRep":
        return GenericRep(rep.algebra(), rep.continuous_generators(), rep.discrete_generators())

    def algebra(self) -> np.ndarray:
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
