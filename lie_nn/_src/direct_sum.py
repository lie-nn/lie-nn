import numpy as np
from multimethod import multimethod

import lie_nn as lie


def direct_sum(*reps) -> lie.Rep:
    assert len(reps) > 0
    if len(reps) == 1:
        return reps[0]
    return _direct_sum(reps[0], direct_sum(*reps[1:]))


def _chk(r1, r2):
    assert np.allclose(r1.A, r2.A, atol=1e-10)
    assert len(r1.H) == len(r2.H)


@multimethod
def _direct_sum(rep1: lie.Rep, rep2: lie.Rep) -> lie.SumRep:
    _chk(rep1, rep2)
    return lie.SumRep((rep1, rep2), force=True)


@multimethod
def _direct_sum(sumrep: lie.SumRep, rep: lie.Rep) -> lie.SumRep:  # noqa: F811
    _chk(sumrep, rep)
    return lie.SumRep(sumrep.reps + (rep,), force=True)


@multimethod
def _direct_sum(rep: lie.Rep, sumrep: lie.SumRep) -> lie.SumRep:  # noqa: F811
    _chk(rep, sumrep)
    return lie.SumRep((rep,) + sumrep.reps, force=True)


@multimethod
def _direct_sum(sumrep1: lie.SumRep, sumrep2: lie.SumRep) -> lie.SumRep:  # noqa: F811
    _chk(sumrep1, sumrep2)
    return lie.SumRep(sumrep1.reps + sumrep2.reps, force=True)


@multimethod
def _direct_sum(qrep: lie.QRep, rep: lie.Rep) -> lie.Rep:  # noqa: F811
    _chk(qrep, rep)
    return lie.change_basis(
        lie.utils.direct_sum(qrep.Q, np.eye(rep.dim)), direct_sum(qrep.rep, rep)
    )


@multimethod
def _direct_sum(rep: lie.Rep, qrep: lie.QRep) -> lie.Rep:  # noqa: F811
    _chk(rep, qrep)
    return lie.change_basis(
        lie.utils.direct_sum(np.eye(rep.dim), qrep.Q), direct_sum(rep, qrep.rep)
    )


@multimethod
def _direct_sum(qrep: lie.QRep, rep: lie.SumRep) -> lie.Rep:  # noqa: F811
    _chk(qrep, rep)
    return lie.change_basis(
        lie.utils.direct_sum(qrep.Q, np.eye(rep.dim)), direct_sum(qrep.rep, rep)
    )


@multimethod
def _direct_sum(rep: lie.SumRep, qrep: lie.QRep) -> lie.Rep:  # noqa: F811
    _chk(rep, qrep)
    return lie.change_basis(
        lie.utils.direct_sum(np.eye(rep.dim), qrep.Q), direct_sum(rep, qrep.rep)
    )
