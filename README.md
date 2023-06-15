# lie-nn
The aim of this library is to help the development of equivariant neural networks on reductive Lie groups. It contains fundamental mathematical operations such as tensor products, symmetric products, irreducible representations.

The library is modular, and new groups can be easily implemented. 

## Installation

To install via `pip`, follow the steps below:

```
git clone git@github.com:lie-nn/lie-nn.git
pip install ./lie-nn
```


## Data Types
The data are organised in three categories:

- `GenericRep` - a generic representation of a Lie group
- `ReducedRep` - a reduced representation into irreps
- `TabulatedIrrep` - a tabulated irreps (`SU2`, `SU2Real`, `O3`, `SO3`, `SL2C`, `SO13`, `SU3`, `SU4`, `Z2`)

The current tabulated irreps include:

|    Groups     | Potential Applications          |
| ------------- | -------------                   |
| $SO(3)$  | 3D Point Clouds (Molecules, Vision)  |
| $SO(1,3)$  | Particles Physics                  |
| $SU(2^{n})$ | Quantum Physics                   |
| $SU(3)$ | QCD                                   | 
| $SL(2, \mathbb{C})$| -                          |
 
 We aim to add new tabulated irreps groups including the other classical Lie groups $Sp(2n)$, $SO(2n+1)$ and $SO(2n)$ (PR are welcomed). 

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
## References

If you use this code, please cite our papers:
```text
@misc{batatia2023general,
      title={A General Framework for Equivariant Neural Networks on Reductive Lie Groups}, 
      author={Ilyes Batatia and Mario Geiger and Jose Munoz and Tess Smidt and Lior Silberman and Christoph Ortner},
      year={2023},
      eprint={2306.00091},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
