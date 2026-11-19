# Examples

These pages contain small but complete applications. Each one is executed when
the documentation is built, and its result is checked against an independent
implementation with NumPy.

| Example                                                        | Grid         | Main features                                         |
| -------------------------------------------------------------- | ------------ | ----------------------------------------------------- |
| [Horizontal diffusion](horizontal-diffusion.md)                | Cartesian    | Composition of operators, `where`, `concat_where`     |
| [Implicit vertical diffusion](vertical-diffusion.md)           | Cartesian    | Scan operators, tridiagonal solver                    |
| [Operators on an unstructured mesh](unstructured-operators.md) | Unstructured | Connectivities, neighbor reductions, local dimensions |

```{toctree}
---
hidden:
---
horizontal-diffusion
vertical-diffusion
unstructured-operators
```
