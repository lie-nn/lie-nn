import numpy as np
from multipledispatch import dispatch

from .multiply import multiply
from .direct_sum import direct_sum
from .change_basis import change_basis
from .rep import GenericRep, MulRep, QRep, Rep, SumRep, TabulatedIrrep


@dispatch(Rep, Rep)
def tensor_product(rep1: Rep, rep2: Rep) -> GenericRep:
    assert np.allclose(rep1.A, rep2.A)  # same lie algebra
    X1, H1, I1 = rep1.X, rep1.H, np.eye(rep1.dim)
    X2, H2, I2 = rep2.X, rep2.H, np.eye(rep2.dim)
    assert H1.shape[0] == H2.shape[0]  # same discrete dimension
    d = rep1.dim * rep2.dim
    return GenericRep(
        A=rep1.A,
        X=(np.einsum("aij,kl->aikjl", X1, I2) + np.einsum("ij,akl->aikjl", I1, X2)).reshape(
            X1.shape[0], d, d
        ),
        H=np.einsum("aij,akl->aikjl", H1, H2).reshape(H1.shape[0], d, d),
    )


@dispatch(TabulatedIrrep, TabulatedIrrep)
def tensor_product(irrep1: TabulatedIrrep, irrep2: TabulatedIrrep) -> Rep:  # noqa: F811
    assert np.allclose(irrep1.A, irrep2.A)  # same lie algebra
    CG_list = []
    irreps_list = []
    for ir_out in irrep1 * irrep2:
        CG = np.moveaxis(
            irrep1.clebsch_gordan(irrep1, irrep2, ir_out), 3, 1
        )  # [sol, ir3, ir1, ir2]
        mul = CG.shape[0]
        CG = CG.reshape(CG.shape[0] * CG.shape[1], CG.shape[2] * CG.shape[3])
        CG_list.append(CG)
        irreps_list.append(multiply(mul, ir_out))
    CG = np.concatenate(CG_list, axis=0)
    Q = np.linalg.inv(CG)
    return change_basis(Q, direct_sum(*irreps_list))


@dispatch(MulRep, Rep)
def tensor_product(mulrep: MulRep, rep: Rep) -> Rep:  # noqa: F811
    assert np.allclose(mulrep.A, rep.A)  # same lie algebra
    return multiply(mulrep.mul, tensor_product(mulrep.rep, rep))


@dispatch(MulRep, MulRep)
def tensor_product(mulrep: MulRep, rep: MulRep) -> Rep:  # noqa: F811
    assert np.allclose(mulrep.A, rep.A)  # same lie algebra
    return multiply(mulrep.mul, tensor_product(mulrep.rep, rep))


@dispatch(Rep, MulRep)
def tensor_product(rep: Rep, mulrep: MulRep) -> Rep:  # noqa: F811
    assert np.allclose(rep.A, mulrep.A)  # same lie algebra

    Q = np.reshape(
        np.einsum(
            "ij,mn,uv->ium vjn",
            np.eye(rep.dim),
            np.eye(mulrep.rep.dim),
            np.eye(mulrep.mul),
        ),
        (
            rep.dim * mulrep.rep.dim * mulrep.mul,
            rep.dim * mulrep.rep.dim * mulrep.mul,
        ),
    )
    return change_basis(Q, multiply(mulrep.mul, tensor_product(rep, mulrep.rep)))


@dispatch(SumRep, Rep)
def tensor_product(sumrep: SumRep, rep: Rep) -> Rep:  # noqa: F811
    return direct_sum(*[tensor_product(subrep, rep) for subrep in sumrep.reps])


@dispatch(SumRep, SumRep)
def tensor_product(sumrep: SumRep, rep: SumRep) -> Rep:  # noqa: F811
    return direct_sum(*[tensor_product(subrep, rep) for subrep in sumrep.reps])


@dispatch(Rep, SumRep)
def tensor_product(rep: Rep, sumrep: SumRep) -> Rep:  # noqa: F811
    list = []
    Q = np.zeros((rep.dim, sumrep.dim, rep.dim * sumrep.dim))
    k = 0
    j = 0
    for subrep in sumrep.reps:
        tp = tensor_product(rep, subrep)
        list += [tp]
        q = np.eye(tp.dim).reshape(rep.dim, subrep.dim, tp.dim)
        Q[:, j : j + subrep.dim, k : k + tp.dim] = q
        k += tp.dim
        j += subrep.dim

    Q = Q.reshape(rep.dim * sumrep.dim, rep.dim * sumrep.dim)
    return change_basis(Q, direct_sum(*list))


@dispatch(QRep, Rep)
def tensor_product(qrep: QRep, rep: Rep) -> Rep:  # noqa: F811
    dim = qrep.dim * rep.dim
    Q = np.einsum("ij,kl->ikjl", qrep.Q, np.eye(rep.dim)).reshape(dim, dim)
    return change_basis(Q, tensor_product(qrep.rep, rep))


@dispatch(Rep, QRep)
def tensor_product(rep: Rep, qrep: QRep) -> Rep:  # noqa: F811
    dim = rep.dim * qrep.dim
    Q = np.einsum("ij,kl->ikjl", np.eye(rep.dim), qrep.Q).reshape(dim, dim)
    return change_basis(Q, tensor_product(rep, qrep.rep))


@dispatch(QRep, QRep)
def tensor_product(qrep1: QRep, qrep2: QRep) -> Rep:  # noqa: F811
    dim = qrep1.dim * qrep2.dim
    Q = np.einsum("ij,kl->ikjl", qrep1.Q, qrep2.Q).reshape(dim, dim)
    return change_basis(Q, tensor_product(qrep1.rep, qrep2.rep))


def tensor_power(rep: Rep, n: int) -> Rep:
    result = None

    while True:
        if n & 1:
            if result is None:
                result = rep
            else:
                result = tensor_product(rep, result)
        n >>= 1

        if n == 0:
            if result is None:
                return rep.create_trivial()
            else:
                return result
        rep = tensor_product(rep, rep)


# @dispatch(ReducedRep, int)
# def tensor_power(rep: ReducedRep, n: int) -> ReducedRep:
#     # TODO reduce into irreps and wrap with the change of basis that
#       maps to the usual tensor product
#     # TODO as well reduce into irreps of S_n
#     # and diagonalize irreps of S_n in the same basis that diagonalizes
#       irreps of S_{n-1} (unclear how to do this)
#     raise NotImplementedError
