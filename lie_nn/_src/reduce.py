import numpy as np
from multimethod import multimethod

import lie_nn as lie


@multimethod
def reduce(rep: lie.ConjRep) -> lie.ReducedRep:  # noqa: F811
    red = reduce(rep.rep)
    reps = tuple((mul, lie.conjugate(ir)) for mul, ir in red.reps)
    return lie.ReducedRep(A=rep.A, num_H=len(rep.H), Q=red.Q.conj(), reps=reps, force=True)


@multimethod
def reduce(rep: lie.MulRep) -> lie.ReducedRep:  # noqa: F811
    red = reduce(rep.rep)
    reps = tuple((rep.mul * mul, ir) for mul, ir in red.reps)
    Q = np.concatenate([np.repeat(q, rep.mul, axis=1) for q in red.split_Q()], axis=1)
    return lie.ReducedRep(A=rep.A, num_H=len(rep.H), Q=Q, reps=reps, force=True)


@multimethod
def reduce(rep: lie.QRep) -> lie.ReducedRep:  # noqa: F811
    red = reduce(rep.rep)
    return lie.ReducedRep(A=rep.A, num_H=len(rep.H), Q=rep.Q @ red.Q, reps=red.reps, force=True)


@multimethod
def reduce(rep: lie.SumRep) -> lie.ReducedRep:  # noqa: F811
    if rep.dim == 0:
        return lie.ReducedRep(
            A=rep.A, num_H=len(rep.H), Q=np.eye(rep.dim), reps=((1, rep),), force=True
        )
    reds = [reduce(subrep) for subrep in rep.reps]
    Q = lie.utils.direct_sum(*[red.Q for red in reds])
    mulirs = sum([red.reps for red in reds], ())
    blocks = []
    i = 0
    for mul, ir in mulirs:
        Qi = Q[:, i : i + mul * ir.dim]
        blocks.append((mul, ir, Qi))
        i += mul * ir.dim

    blocks.sort(key=lambda x: x[1].dim)
    merged_blocks = []

    while len(blocks) > 0:
        mul, ir, Qi = blocks.pop(0)
        j = len(blocks) - 1
        while j >= 0:
            mul2, ir2, Qi2 = blocks[j]
            if lie.are_isomorphic(ir, ir2):
                mul += mul2
                q = lie.infer_change_of_basis(ir, ir2)  # q ir = ir2 q
                assert len(q) == 1
                Qi = np.concatenate((Qi, Qi2 @ q[0]), axis=1)
                blocks.pop(j)
            j -= 1
        merged_blocks.append((mul, ir, Qi))

    Q = np.concatenate([Qi for _, _, Qi in merged_blocks], axis=1)
    mulirs = tuple((mul, ir) for mul, ir, _ in merged_blocks)
    return lie.ReducedRep(A=rep.A, num_H=len(rep.H), Q=Q, reps=mulirs, force=True)


@multimethod
def reduce(rep: lie.Irrep) -> lie.ReducedRep:  # noqa: F811
    return lie.ReducedRep(
        A=rep.A, num_H=len(rep.H), Q=np.eye(rep.dim), reps=((1, rep),), force=True
    )


@multimethod
def reduce(rep: lie.Rep) -> lie.ReducedRep:  # noqa: F811
    r"""Reduce an unknown representation to a reduced form.
    This operation is slow and should be avoided if possible.
    """

    def try_reduce():
        Ys = lie.utils.decompose_rep_into_irreps(np.concatenate([rep.X, rep.H]))
        d = rep.lie_dim
        Qs = []
        irs = []
        for mul, Y in Ys:
            ir = lie.GenericRep(rep.A, Y[:d], Y[d:])
            Q = lie.infer_change_of_basis(ir, rep)
            if len(Q) != mul:
                return None

            Q = np.einsum("mij->imj", Q).reshape((rep.dim, mul * ir.dim))
            Qs.append(Q)
            mul_ir = (mul, ir)
            irs.append(mul_ir)

        Q = np.concatenate(Qs, axis=1)
        if np.allclose(Q, np.eye(rep.dim), atol=1e-10):
            Q = np.eye(rep.dim)

        return lie.ReducedRep(A=rep.A, num_H=len(rep.H), Q=Q, reps=tuple(irs), force=True)

    import time

    t = time.time()
    for n in range(100):
        red = try_reduce()
        if red is not None:
            return red
        if time.time() - t > 1:
            break

    raise ValueError(
        f"Could not reduce representation after {time.time() - t} seconds and {n} tries."
    )
