# lie-nn
This library aims to help the development of equivariant polynomials on reductive Lie groups and finite groups. It contains fundamental mathematical operations such as tensor products, symmetric products, and direct sums of reducible and irreducible representations of groups.

The library is modular, and new groups can be easily implemented.

## Installation

To install via `pip`, follow the steps below:

```
git clone git@github.com:lie-nn/lie-nn.git
pip install ./lie-nn
```

## Tabulated Irreducible Representations

Irreducible representations (irreps) of reductive Lie groups play an essential role as they are budling blocks of other finite-dimensional representations.
We implement irreducible representations for a variety of Lie groups by providing the following:
- labelling of the irreps
- their dimensions
- an explicit basis (called generators)
- direct sums of irreps
- decompositions of tensor product of irreps into irreps (**Clebsch Gordans**)
- symmetric power of irreps (Symmetric generalized Clebsch **Clebsch Gordans**)

`lie-nn` currently has tabulated the irreps of the following groups:

|    Groups     | Potential Applications          |
| ------------- | -------------                   |
| $\mathrm{SO}_{\mathbb{R}}(3)$  | 3D Point Clouds (Molecules, Vision)  |
| $\mathrm{O}_{\mathbb{R}}(3)$   | Rotations + Reflections              |
| $\mathrm{SO}_{\mathbb{R}}(1,3)$  | Particles Physics                  |
| $\mathrm{SU}_{\mathbb{R}}(N)$ | Quantum Physics                   |
| $\mathrm{SU}_{\mathbb{R}}(3)$ | QCD                                   | 
| $\mathrm{U}_{\mathbb{R}}(1)$  | Electromagnetism                      |
| $\mathrm{SL}_{\mathbb{R}}(2, \mathbb{C})$| -                          |
 
Irreps of a product of all these groups are also supported. Moreover isomorphics groups have the same irreducible representations. As all the irreps of $\mathfrak{sl}_ {n}$ ($SU(N)$ in the code) are implemented, the irreps of the following Lie algebras can also be obtained:

$\mathfrak{so}_ {3} \simeq \mathfrak{sp}_ {2}  = \mathfrak{sl}_ {2}$

$\mathfrak{so}_ {4} \simeq \mathfrak{sl}_ {2} \oplus \mathfrak{sl} _{2}$

$\mathfrak{so}_ {6} \simeq \mathfrak{sl}_ {4}$
 
We aim to add new tabulated irreps for more groups, including the other classical complex Lie groups $Sp(2n)$, $SO(2n+1)$ and $SO(2n)$ (PR are welcomed). 
 
## Data Types
The data are organised in three categories:

- `GenericRep` - a generic representation of a Lie group
- `ReducedRep` - a reduced representation into irreps
- `Rep` - a reducible representations (`Sn`, `so(2n)_adjoint`, `so(2n+1)_adjoint`)
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
