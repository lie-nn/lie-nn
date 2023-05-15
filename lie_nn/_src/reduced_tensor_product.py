"""
History of the different versions of the code:
- Initially developed by Mario Geiger in `e3nn`
- Ported in julia by Song Kim https://github.com/songk42/ReducedTensorProduct.jl
- Ported in `e3nn-jax` by Mario Geiger
- Ported in `lie_nn` by Mario Geiger and Ilyes Batatia
"""
import functools
import itertools
from typing import FrozenSet, List, Optional, Tuple, Union

from lie_nn import TabulatedIrrep, ReducedRep, MulIrrep, Rep
import numpy as np
import lie_nn._src.discrete_groups.perm as perm
from .util import basis_intersection, round_to_sqrt_rational, prod
from typing import NamedTuple


class RepArray(NamedTuple):
    rep: Rep
    array: np.ndarray
    list: List[np.ndarray]


class IrrepsArray:
    irreps: Tuple[MulIrrep, ...]
    list: List[np.ndarray]

    @property
    def array(self):
        return np.concatenate([np.reshape(x, x.shape[:-2] + (-1,)) for x in self.list], axis=-1)

    def __init__(self, *, irreps, list):
        assert len(irreps) == len(list)
        shapes = []
        for mul_ir, x in zip(irreps, list):
            assert x.shape[-2] == mul_ir.mul
            assert x.shape[-1] == mul_ir.rep.dim
            shapes.append(x.shape[:-2])
        assert len(set(shapes)) == 1
        self.irreps = tuple(irreps)
        self.list = list

    def sorted(self):
        indices = list(range(len(self.irreps)))
        indices = sorted(indices, key=lambda i: (self.irreps[i].rep, self.irreps[i].mul))
        return IrrepsArray(
            list=[self.list[i] for i in indices],
            irreps=tuple(self.irreps[i] for i in indices),
        )

    def simplify(self):
        muls = []
        irreps = []
        list = []

        for i, mul_irrep in enumerate(self.irreps):
            mul, irrep = mul_irrep.mul, mul_irrep.rep
            x = self.list[i]

            if i == 0 or irrep != irreps[-1]:
                irreps.append(irrep)
                list.append(x)
                muls.append(mul)
            else:
                list[-1] = np.concatenate([list[-1], x], axis=-2)
                muls[-1] += mul
        return IrrepsArray(
            list=list, irreps=tuple(MulIrrep(mul, irrep) for mul, irrep in zip(muls, irreps))
        )

    def reshape(self, shape):
        assert shape[-1] == -1 or shape[-1] == sum(mul_irrep.mul for mul_irrep in self.irreps)

        x_list = []
        for mul_irrep, x in zip(self.irreps, self.list):
            mul, irrep = mul_irrep.mul, mul_irrep.rep
            x_list.append(x.reshape(shape[:-1] + (mul, irrep.dim)))

        return IrrepsArray(list=x_list, irreps=self.irreps)


def _to_reducedrep(irreps) -> ReducedRep:
    if isinstance(irreps, TabulatedIrrep):
        irreps = MulIrrep(1, irreps)
    if isinstance(irreps, MulIrrep):
        irreps = ReducedRep.from_irreps([irreps])
    assert isinstance(irreps, ReducedRep)
    return irreps


def reduced_tensor_product_basis(
    formula_or_irreps_list: Union[str, List[ReducedRep]],
    *,
    epsilon: float = 1e-5,
    **irreps_dict,
) -> RepArray:
    r"""Reduce a tensor product of multiple irreps subject
    to some permutation symmetry given by a formula.

    Args:
        formula_or_irreps_list (str or list of Irreps): a formula
            of the form ``ijk=jik=ikj`` or ``ijk=-jki``.
            The left hand side is the original formula and the right hand side are
            the signed permutations.
            If no index symmetry is present, a list of irreps can be given instead.

        epsilon (float): the tolerance for the Gram-Schmidt orthogonalization. Default: ``1e-5``
        irreps_dict (dict): the irreps of each index of the formula. For instance ``i="1x1o"``.

    Returns:
        RepArray: The change of basis
            The shape is ``(d1, ..., dn, irreps_out.dim)``
            where ``di`` is the dimension of the index ``i`` and ``n``
            is the number of indices in the formula.
    """

    if isinstance(formula_or_irreps_list, (tuple, list)):
        irreps_list = formula_or_irreps_list
        irreps_tuple = tuple(_to_reducedrep(irreps) for irreps in irreps_list)
        formulas: FrozenSet[Tuple[int, Tuple[int, ...]]] = frozenset(
            {(1, tuple(range(len(irreps_tuple))))}
        )
        out = _reduced_tensor_product_basis(irreps_tuple, formulas, epsilon)
        return RepArray(ReducedRep.from_irreps(out.irreps), out.array, out.list)

    formula = formula_or_irreps_list
    f0, perm_repr = germinate_perm_repr(formula)

    irreps_dict = {i: _to_reducedrep(irs) for i, irs in irreps_dict.items()}

    for i in irreps_dict:
        if len(i) != 1:
            raise TypeError(f"got an unexpected keyword argument '{i}'")

    for _sign, p in perm_repr:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in irreps_dict and j in irreps_dict and irreps_dict[i] != irreps_dict[j]:
                raise RuntimeError(f"irreps of {i} and {j} should be the same")
            if i in irreps_dict:
                irreps_dict[j] = irreps_dict[i]
            if j in irreps_dict:
                irreps_dict[i] = irreps_dict[j]

    for i in f0:
        if i not in irreps_dict:
            raise RuntimeError(f"index {i} has no irreps associated to it")

    for i in irreps_dict:
        if i not in f0:
            raise RuntimeError(f"index {i} has an irreps but does not appear in the fomula")

    irreps_tuple = tuple(irreps_dict[i] for i in f0)

    out = _reduced_tensor_product_basis(irreps_tuple, perm_repr, epsilon)
    return RepArray(ReducedRep.from_irreps(out.irreps), out.array, out.list)


def reduced_symmetric_tensor_product_basis(
    irreps: ReducedRep,
    order: int,
    *,
    epsilon: float = 1e-5,
) -> RepArray:
    r"""Reduce a symmetric tensor product.

    Args:
        irreps (Irreps): the irreps of each index.
        order (int): the order of the tensor product. i.e. the number of indices.

    Returns:
        RepArray: The change of basis
            The shape is ``(d, ..., d, irreps_out.dim)``
            where ``d`` is the dimension of ``irreps``.
    """
    irreps = _to_reducedrep(irreps)
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]] = frozenset(
        (1, p) for p in itertools.permutations(range(order))
    )
    out = _reduced_tensor_product_basis(tuple([irreps] * order), perm_repr, epsilon)
    return RepArray(ReducedRep.from_irreps(out.irreps), out.array, out.list)


# @functools.lru_cache(maxsize=None)
def _reduced_tensor_product_basis(
    irreps_tuple: Tuple[ReducedRep, ...],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    epsilon: float,
) -> IrrepsArray:
    dims = tuple(irps.dim for irps in irreps_tuple)

    def get_initial_basis(reduced_rep: ReducedRep, i: int) -> List[np.ndarray]:
        x = np.reshape(
            np.eye(reduced_rep.dim) if reduced_rep.Q is None else np.linalg.inv(reduced_rep.Q).T,
            (1,) * i + (reduced_rep.dim,) + (1,) * (len(irreps_tuple) - i - 1) + (reduced_rep.dim,),
        )
        x_list = []
        cursor = 0
        for mul_ir in reduced_rep.irreps:
            mul, ir = mul_ir.mul, mul_ir.rep
            x_list.append(
                x[..., cursor : cursor + mul * ir.dim].reshape(x.shape[:-1] + (mul, ir.dim))
            )
            cursor += mul * ir.dim
        return x_list

    bases = [
        (
            frozenset({i}),
            IrrepsArray(list=get_initial_basis(reduced_rep, i), irreps=reduced_rep.irreps),
        )
        for i, reduced_rep in enumerate(irreps_tuple)
    ]

    while True:
        if len(bases) == 1:
            f, b = bases[0]
            assert f == frozenset(range(len(irreps_tuple)))
            return b.sorted().simplify()

        if len(bases) == 2:
            (fa, a) = bases[0]
            (fb, b) = bases[1]
            f = frozenset(fa | fb)
            ab = reduce_basis_product(a, b)
            if len(subrepr_permutation(f, perm_repr)) == 1:
                return ab.sorted().simplify()
            p = reduce_subgroup_permutation(f, perm_repr, dims)
            ab = constrain_rotation_basis_by_permutation_basis(
                ab, p, epsilon=epsilon, round_fn=round_to_sqrt_rational
            )
            return ab.sorted().simplify()

        # greedy algorithm
        min_p = np.inf
        best = None

        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                (fa, _) = bases[i]
                (fb, _) = bases[j]
                f = frozenset(fa | fb)
                p_dim = reduce_subgroup_permutation(f, perm_repr, dims, return_dim=True)
                if p_dim < min_p:
                    min_p = p_dim
                    best = (i, j, f)

        i, j, f = best
        del bases[j]
        del bases[i]
        sub_irreps = tuple(irreps_tuple[i] for i in f)
        sub_perm_repr = subrepr_permutation(f, perm_repr)
        ab = _reduced_tensor_product_basis(sub_irreps, sub_perm_repr, epsilon)
        ab = ab.reshape(tuple(dims[i] if i in f else 1 for i in range(len(dims))) + (-1,))
        bases = [(f, ab)] + bases


@functools.lru_cache(maxsize=None)
def germinate_perm_repr(formula: str) -> Tuple[str, FrozenSet[Tuple[int, Tuple[int, ...]]]]:
    """Convert the formula (generators) into a group."""
    formulas = [(-1 if f.startswith("-") else 1, f.replace("-", "")) for f in formula.split("=")]
    s0, f0 = formulas[0]
    assert s0 == 1

    for _s, f in formulas:
        if len(set(f)) != len(f) or set(f) != set(f0):
            raise RuntimeError(f"{f} is not a permutation of {f0}")
        if len(f0) != len(f):
            raise RuntimeError(f"{f0} and {f} don't have the same number of indices")

    # `perm_repr` is a list of (sign, permutation of indices)
    # each formula can be viewed as a permutation of the original formula
    perm_repr = {
        (s, tuple(f.index(i) for i in f0)) for s, f in formulas
    }  # set of generators (permutations)

    # they can be composed, for instance if you have ijk=jik=ikj
    # you also have ijk=jki
    # applying all possible compositions creates an entire group
    while True:
        n = len(perm_repr)
        perm_repr = perm_repr.union([(s, perm.inverse(p)) for s, p in perm_repr])
        perm_repr = perm_repr.union(
            [(s1 * s2, perm.compose(p1, p2)) for s1, p1 in perm_repr for s2, p2 in perm_repr]
        )
        if len(perm_repr) == n:
            break  # we break when the set is stable => it is now a group \o/

    return f0, frozenset(perm_repr)


def reduce_basis_product(
    basis1: IrrepsArray,
    basis2: IrrepsArray,
    filter_ir_out: Optional[List[TabulatedIrrep]] = None,
) -> IrrepsArray:
    """Reduce the product of two basis."""
    basis1 = basis1.sorted().simplify()
    basis2 = basis2.sorted().simplify()

    new_irreps: List[Tuple[int, TabulatedIrrep]] = []
    new_list: List[np.ndarray] = []

    for mul_ir1, x1 in zip(basis1.irreps, basis1.list):
        mul1, ir1 = mul_ir1.mul, mul_ir1.rep
        for mul_ir2, x2 in zip(basis2.irreps, basis2.list):
            mul2, ir2 = mul_ir2.mul, mul_ir2.rep
            for ir in ir1 * ir2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                cg = ir.clebsch_gordan(ir1, ir2, ir)
                x = np.einsum(
                    "...ui,...vj,wijk->...wuvk",
                    x1,
                    x2,
                    cg,
                )
                x = np.reshape(x, x.shape[:-4] + (cg.shape[0] * mul1 * mul2, ir.dim))
                new_irreps.append((cg.shape[0] * mul1 * mul2, ir))
                new_list.append(x)

    new = IrrepsArray(irreps=tuple(MulIrrep(mul, ir) for mul, ir in new_irreps), list=new_list)
    return new.sorted().simplify()


def constrain_rotation_basis_by_permutation_basis(
    rotation_basis: IrrepsArray,
    permutation_basis: np.ndarray,
    *,
    epsilon=1e-5,
    round_fn=lambda x: x,
) -> IrrepsArray:
    """Constrain a rotation basis by a permutation basis.

    Args:
        rotation_basis (e3nn.IrrepsArray): A rotation basis
        permutation_basis (np.ndarray): A permutation basis

    Returns:
        e3nn.IrrepsArray: A rotation basis that is constrained by the permutation basis.
    """
    assert all(x.shape[:-2] == permutation_basis.shape[1:] for x in rotation_basis.list)

    perm = np.reshape(permutation_basis, (permutation_basis.shape[0], -1))  # (free, dim)

    new_irreps: List[Tuple[int, TabulatedIrrep]] = []
    new_list: List[np.ndarray] = []

    for rotation_basis_mul_ir, rot_basis in zip(rotation_basis.irreps, rotation_basis.list):
        mul, ir = rotation_basis_mul_ir.mul, rotation_basis_mul_ir.rep
        R = rot_basis[..., 0]
        R = np.reshape(R, (-1, mul)).T  # (mul, dim)

        perm_opt = perm[~np.all(perm[:, ~np.all(R == 0, axis=0)] == 0, axis=1)]
        P, _ = basis_intersection(R, perm_opt, epsilon=epsilon, round_fn=round_fn)

        if P.shape[0] > 0:
            new_irreps.append((P.shape[0], ir))
            new_list.append(np.einsum("vu,...ui->...vi", P, rot_basis))

    return IrrepsArray(irreps=tuple(MulIrrep(mul, ir) for mul, ir in new_irreps), list=new_list)


def subrepr_permutation(
    sub_f0: FrozenSet[int], perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]]
) -> FrozenSet[Tuple[int, Tuple[int, ...]]]:
    sor = sorted(sub_f0)
    return frozenset(
        {
            (s, tuple(sor.index(i) for i in p if i in sub_f0))
            for s, p in perm_repr
            if all(i in sub_f0 or i == j for j, i in enumerate(p))
        }
    )


def reduce_subgroup_permutation(
    sub_f0: FrozenSet[int],
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]],
    dims: Tuple[int, ...],
    return_dim: bool = False,
) -> np.ndarray:
    sub_perm_repr = subrepr_permutation(sub_f0, perm_repr)
    sub_dims = tuple(dims[i] for i in sub_f0)
    if len(sub_perm_repr) == 1:
        if return_dim:
            return prod(sub_dims)
        return np.eye(prod(sub_dims)).reshape((prod(sub_dims),) + sub_dims)
    base = reduce_permutation_base(sub_perm_repr, sub_dims)
    if return_dim:
        return len(base)
    permutation_basis = reduce_permutation_matrix(base, sub_dims)
    return np.reshape(
        permutation_basis, (-1,) + tuple(dims[i] if i in sub_f0 else 1 for i in range(len(dims)))
    )


@functools.lru_cache(maxsize=None)
def full_base_fn(dims: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return list(itertools.product(*(range(d) for d in dims)))


@functools.lru_cache(maxsize=None)
def reduce_permutation_base(
    perm_repr: FrozenSet[Tuple[int, Tuple[int, ...]]], dims: Tuple[int, ...]
) -> FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]]:
    full_base = full_base_fn(dims)  # (0, 0, 0), (0, 0, 1), (0, 0, 2), ... (3, 3, 3)
    # len(full_base) degrees of freedom in an unconstrained tensor

    # but there is constraints given by the group `formulas`
    # For instance if `ij=-ji`, then 00=-00, 01=-01 and so on
    base = set()
    for x in full_base:
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = {(s, tuple(x[i] for i in p)) for s, p in perm_repr}
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if not (-1, x) in xs:
            # the sign is arbitrary, put both possibilities
            base.add(frozenset({frozenset(xs), frozenset({(-s, x) for s, x in xs})}))

    # len(base) is the number of degrees of freedom in the tensor.

    return frozenset(base)


@functools.lru_cache(maxsize=None)
def reduce_permutation_matrix(
    base: FrozenSet[FrozenSet[FrozenSet[Tuple[int, Tuple[int, ...]]]]], dims: Tuple[int, ...]
) -> np.ndarray:
    base = sorted(
        [sorted([sorted(xs) for xs in x]) for x in base]
    )  # requested for python 3.7 but not for 3.8 (probably a bug in 3.7)

    # First we compute the change of basis (projection) between full_base and base
    d_sym = len(base)
    Q = np.zeros((d_sym, prod(dims)))

    for i, x in enumerate(base):
        x = max(x, key=lambda xs: sum(s for s, x in xs))
        for s, e in x:
            j = 0
            for k, d in zip(e, dims):
                j *= d
                j += k
            Q[i, j] = s / len(x) ** 0.5

    np.testing.assert_allclose(Q @ Q.T, np.eye(d_sym))

    return Q.reshape(d_sym, *dims)
