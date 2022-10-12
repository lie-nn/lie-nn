import dataclasses

import numpy as np


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

    def __repr__(self) -> str:
        return f"Rep(dim={self.dim}, lie_dim={self.lie_dim}, len(H)={len(self.discrete_generators())})"

    def is_trivial(self) -> bool:
        return self.dim == 1 and np.all(self.continuous_generators() == 0.0) and np.all(self.discrete_generators() == 1.0)


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
