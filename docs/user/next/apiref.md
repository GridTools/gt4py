# API reference

Reference of the public API of `gt4py.next`, that is, the names exported by the
`gt4py.next` package itself. It is generated from the docstrings in the source
code. Everything else in the package is an implementation detail which may
change without notice.

The usual way to access the API is through the `gtx` alias:

```python
import gt4py.next as gtx
```

```{eval-rst}
.. currentmodule:: gt4py.next
```

## Dimensions and domains

`Dims` is only used in type annotations, to state the dimensions of a field:
`gtx.Field[gtx.Dims[IDim, KDim], gtx.float64]`.

```{eval-rst}
.. autoclass:: Dimension
   :members:

.. autoclass:: DimensionKind
   :members:
   :undoc-members:

.. autoclass:: UnitRange
   :members:

.. autofunction:: unit_range

.. autoclass:: Domain
   :members:

.. autofunction:: domain

.. autoclass:: GridType
   :members:
   :undoc-members:

.. autofunction:: is_staggered

.. autofunction:: flip_staggered

.. autofunction:: as_non_staggered
```

## Fields

```{eval-rst}
.. autoclass:: Field
   :members:

.. autofunction:: as_field

.. autofunction:: empty

.. autofunction:: zeros

.. autofunction:: ones

.. autofunction:: full

.. autoclass:: FieldConstructor
   :members:
```

## Connectivities and offsets

```{eval-rst}
.. autoclass:: Connectivity
   :members:

.. autoclass:: CartesianConnectivity
   :members:

.. autofunction:: as_connectivity

.. autoclass:: FieldOffset
   :members:
```

## Field operators and programs

```{eval-rst}
.. autofunction:: field_operator

.. autofunction:: scan_operator

.. autofunction:: program

.. autofunction:: wait_for_compilation
```

## Built-in functions

Functions which can be called inside field operators. They are also callable
from regular Python code, where they are executed directly on fields.

### Field built-ins

| Function                               | Description                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------------- |
| `where(mask, true_field, false_field)` | Select values from `true_field` where `mask` is true and from `false_field` otherwise. |
| `astype(value, type_)`                 | Convert a field, a scalar or a tuple of them to another data type.                     |
| `broadcast(field, dims)`               | Extend a field or scalar to the given tuple of dimensions.                             |
| `neighbor_sum(field, axis=...)`        | Sum the values of a field along the local dimension `axis`.                            |
| `max_over(field, axis=...)`            | Maximum of the values of a field along the local dimension `axis`.                     |
| `min_over(field, axis=...)`            | Minimum of the values of a field along the local dimension `axis`.                     |

### Math built-ins

All of them work element-wise on fields and scalars.

| Kind                                        | Functions                                                                                                                                                                   |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unary, on any number                        | `abs`, `neg`                                                                                                                                                                |
| Unary, on floating point numbers            | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`, `sqrt`, `exp`, `log`, `gamma`, `cbrt`, `floor`, `ceil`, `trunc` |
| Unary predicates, on floating point numbers | `isfinite`, `isinf`, `isnan`                                                                                                                                                |
| Binary, on any number                       | `minimum`, `maximum`, `fmod`, `power`                                                                                                                                       |

## Data types

Scalar types which can be used in the type annotations of field operators and
programs, and as constructors to convert a value to that type:

| Kind             | Types                                                                    |
| ---------------- | ------------------------------------------------------------------------ |
| Boolean          | `bool`                                                                   |
| Signed integer   | `int8`, `int16`, `int32`, `int64`, `int`, `IndexType` (alias of `int32`) |
| Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64`                                    |
| Floating point   | `float32`, `float64`, `float`                                            |

`tuple` is also exported to be used in the type annotations of arguments and
return values made of several fields or scalars.

## Backends

Instances which can be passed as the `backend` argument of the decorators.
Without a backend, field operators and programs are executed directly in
Python (_embedded_ execution).

| Backend       | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `gtfn_cpu`    | Generates and compiles GridTools C++ code for CPUs.              |
| `gtfn_gpu`    | Generates and compiles GridTools C++ code for GPUs.              |
| `itir_python` | Generates pure Python code from the intermediate representation. |

## Devices

```{eval-rst}
.. autoclass:: Device
   :members:

.. autoclass:: DeviceType
   :members:
   :undoc-members:

.. data:: CUPY_DEVICE_TYPE

   Type of the GPU accelerator device (``DeviceType.CUDA`` or ``DeviceType.ROCM``),
   if ``cupy`` is installed, or ``None`` otherwise.

.. autofunction:: is_scalar_type
```
