__version__ = "0.0.0"

from ._src.rep import Rep, GenericRep, check_representation_triplet
from ._src.irrep import TabulatedIrrep
from ._src.reduced_rep import MulIrrep, ReducedRep
from ._src.change_basis import change_basis
from ._src.reduce import reduce
from ._src.change_algebra import change_algebra
from ._src.tensor_product import tensor_product, tensor_power
from ._src.direct_sum import direct_sum
from ._src.reduced_tensor_product import (
    reduced_tensor_product_basis,  # TODO: find a better API
    reduced_symmetric_tensor_product_basis,
)
from ._src.clebsch_gordan import clebsch_gordan
from ._src.infer_change_of_basis import infer_change_of_basis
from ._src.conjugate import conjugate
from ._src.real import make_explicitly_real, is_real
from ._src.properties import is_unitary
from ._src.group_product import group_product
from ._src.is_irreducible import is_irreducible

from lie_nn import irreps as irreps
from lie_nn import util as util
from lie_nn import finite as finite

__all__ = [
    "Rep",
    "GenericRep",
    "check_representation_triplet",
    "TabulatedIrrep",
    "MulIrrep",
    "ReducedRep",
    "change_basis",
    "reduce",
    "change_algebra",
    "tensor_product",
    "tensor_power",
    "direct_sum",
    "reduced_tensor_product_basis",
    "reduced_symmetric_tensor_product_basis",
    "clebsch_gordan",
    "infer_change_of_basis",
    "conjugate",
    "make_explicitly_real",
    "is_real",
    "is_unitary",
    "group_product",
    "is_irreducible",
    "irreps",
    "util",
    "finite",
]
