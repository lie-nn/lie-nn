import numpy as np
from multimethod import multimethod

from .change_basis import change_basis
from .conjugate import conjugate
from .direct_sum import direct_sum
from .infer_change_of_basis import infer_change_of_basis
from .multiply import multiply
from .rep import ConjRep, GenericRep, Irrep, MulRep, QRep, Rep, SumRep
from .util import decompose_rep_into_irreps
from .tensor_product import tensor_product


@multimethod
def symmetric_tensor_power(rep: QRep, n: int) -> Rep:  # noqa: F811
    # Q = tensorpower(rep.Q, n)
    # out = P @ Q @ tensorpower(rep, n) @ Q^-1 @ P^-1
    # out = Q' @ P' @ tensorpower(rep, n) ...
    # out = Q' @ Q'' @ symm_tensor_power(rep, n) ...
    raise NotImplementedError


@multimethod
def symmetric_tensor_power(rep: SumRep, n: int) -> Rep:  # noqa: F811
    # for all subreps in rep.reps
    # run symmetric_tensor_power(subrep, i) for i = 1, ..., n
    raise NotImplementedError


@multimethod
def symmetric_tensor_power(rep: MulRep, n: int) -> Rep:  # noqa: F811
    stp = [symmetric_tensor_power(rep.rep, i) for i in range(0, n + 1)]
    # i j if rep.mul == 2 and n == 3
    # 3 0
    # 2 1
    # 1 2
    # 0 3
    raise NotImplementedError


@multimethod
def symmetric_tensor_power(rep: ConjRep, n: int) -> Rep:  # noqa: F811
    return conjugate(symmetric_tensor_power(rep.rep, n))
