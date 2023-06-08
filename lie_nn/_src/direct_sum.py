import numpy as np
from multipledispatch import dispatch

from .rep import Rep, SumRep, QRep
from .util import direct_sum as ds
from .change_basis import change_basis


def direct_sum(*reps) -> Rep:
    assert len(reps) > 0
    if len(reps) == 1:
        return reps[0]
    return _direct_sum(reps[0], direct_sum(*reps[1:]))


def _chk(r1, r2):
    assert np.allclose(r1.A, r2.A, atol=1e-10)
    assert len(r1.H) == len(r2.H)


@dispatch(Rep, Rep)
def _direct_sum(rep1: Rep, rep2: Rep) -> SumRep:
    _chk(rep1, rep2)
    return SumRep((rep1, rep2))


@dispatch(SumRep, Rep)
def _direct_sum(sumrep: SumRep, rep: Rep) -> SumRep:  # noqa: F811
    _chk(sumrep, rep)
    return SumRep(sumrep.reps + (rep,))


@dispatch(Rep, SumRep)
def _direct_sum(rep: Rep, sumrep: SumRep) -> SumRep:  # noqa: F811
    _chk(rep, sumrep)
    return SumRep((rep,) + sumrep.reps)


@dispatch(SumRep, SumRep)
def _direct_sum(sumrep1: SumRep, sumrep2: SumRep) -> SumRep:  # noqa: F811
    _chk(sumrep1, sumrep2)
    return SumRep(sumrep1.reps + sumrep2.reps)


@dispatch(QRep, Rep)
def _direct_sum(qrep: QRep, rep: Rep) -> QRep:  # noqa: F811
    _chk(qrep, rep)
    return change_basis(ds(qrep.Q, np.eye(rep.dim)), direct_sum(qrep.rep, rep))


@dispatch(Rep, QRep)
def _direct_sum(rep: Rep, qrep: QRep) -> QRep:  # noqa: F811
    _chk(rep, qrep)
    return change_basis(ds(np.eye(rep.dim), qrep.Q), direct_sum(rep, qrep.rep))
