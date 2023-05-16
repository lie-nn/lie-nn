# lie-nn

## Data Types

- `GenericRep` - a generic representation of a Lie group
- `ReducedRep` - a reduced representation into irreps
- `TabulatedIrrep` - a tabulated irreps (`SU2`, `SU2Real`, `O3`, `SO3`, `SL2C`, `SO13`, `SU3`, `SU4`, `Z2`)

## Functions

- `change_basis` - change of basis for the representation
- `change_algebra` - change of basis for the algebra
- `infer_change_of_basis` - infer the change of basis matrix between two representations
- `clebsch_gordan` - Clebsch-Gordan coefficients
- `reduce` - creates a reduced representation (`ReducedRep`)
- `direct_sum` - direct sum of two representations

- `tensor_product` - tensor product of two representations
- `reduced_tensor_product_basis` - change of basis matrix between the tensor product and the irreps
- `reduced_symmetric_tensor_product_basis` - change of basis matrix between the symmetric tensor product and the irreps

- `conjugate` - conjugate representation
- `make_explicitly_real` - some representations can be made explicitly real

- `group_product` - representation in the direct product of the groups


## Example

```python
import lie_nn as lie

lie.clebsch_gordan(
    lie.irreps.SU3((1, 1, 0)),
    lie.irreps.SU3((1, 0, 0)),
    lie.irreps.SU3((2, 1, 0)),
)
```

## Formatting the code

```
pycln .
black .
```
