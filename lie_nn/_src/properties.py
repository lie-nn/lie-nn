import numpy as np
from multimethod import multimethod


import lie_nn as lie

# is_irreducible:


@multimethod
def is_irreducible(rep: lie.Rep, *, epsilon: float = 1e-10) -> bool:
    """Returns True if the representation is irreducible."""
    return lie.utils.is_irreducible(np.concatenate([rep.X, rep.H], axis=0), epsilon=epsilon)


@multimethod
def is_irreducible(rep: lie.ConjRep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return is_irreducible(rep.rep, epsilon=epsilon)


@multimethod
def is_irreducible(rep: lie.MulRep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return is_irreducible(rep.rep, epsilon=epsilon)


@multimethod
def is_irreducible(rep: lie.SumRep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return all(is_irreducible(subrep, epsilon=epsilon) for subrep in rep.reps)


@multimethod
def is_irreducible(rep: lie.QRep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return is_irreducible(rep.rep, epsilon=epsilon)


@multimethod
def is_irreducible(rep: lie.Irrep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return True


# is_unitary:


@multimethod
def is_unitary(rep: lie.Rep, *, epsilon: float = 1e-10) -> bool:
    X = rep.continuous_generators()
    H = rep.discrete_generators()
    # exp(X) @ exp(X^H) = 1
    # X + X^H = 0
    H_unit = np.allclose(H @ np.conj(np.transpose(H, (0, 2, 1))), np.eye(rep.dim), atol=epsilon)
    X_unit = np.allclose(X + np.conj(np.transpose(X, (0, 2, 1))), 0, atol=epsilon)
    return H_unit and X_unit


@multimethod
def is_unitary(rep: lie.SumRep) -> bool:  # noqa: F811
    return all(is_unitary(subrep) for subrep in rep.reps)


@multimethod
def is_unitary(rep: lie.MulRep) -> bool:  # noqa: F811
    return is_unitary(rep.rep)


# are_isomorphic:


@multimethod
def are_isomorphic(rep1: lie.Rep, rep2: lie.Rep, *, epsilon: float = 1e-10) -> bool:
    if isinstance(rep1, lie.MulRep) and isinstance(rep2, lie.MulRep) and rep1.mul == rep2.mul:
        return are_isomorphic(rep1.rep, rep2.rep, epsilon=epsilon)

    return lie.utils.are_isomorphic(
        np.concatenate([rep1.X, rep1.H], axis=0),
        np.concatenate([rep2.X, rep2.H], axis=0),
        epsilon=epsilon,
    )


@multimethod
def are_isomorphic(  # noqa: F811
    rep1: lie.ConjRep, rep2: lie.ConjRep, *, epsilon: float = 1e-10
) -> bool:
    return are_isomorphic(rep1.rep, rep2.rep, epsilon=epsilon)


@multimethod
def are_isomorphic(rep1: lie.QRep, rep2: lie.Rep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return are_isomorphic(rep1.rep, rep2, epsilon=epsilon)


@multimethod
def are_isomorphic(rep1: lie.Rep, rep2: lie.QRep, *, epsilon: float = 1e-10) -> bool:  # noqa: F811
    return are_isomorphic(rep1, rep2.rep, epsilon=epsilon)
