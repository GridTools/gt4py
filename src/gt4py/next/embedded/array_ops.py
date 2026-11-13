# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Array operations of the embedded fields, per array namespace.

`ArrayOps` implements everything `NdArrayField` does on its buffer against the
Array API standard, through the namespace `array_api_compat` returns for the
buffer. Any array library conforming to revision 2024.12 or later of the standard
works without registration; earlier revisions do not take Python scalars as
arguments of elementwise functions, which `ArrayOps` passes for scalar operands.

A library that needs different behaviour, or has a faster path outside the
standard, subclasses `ArrayOps`, overrides only those methods and registers an
instance for its compat namespace with `register`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, ClassVar, Final, Optional

import array_api_compat
import array_api_compat.numpy as compat_numpy
import numpy as np

from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
from gt4py.next import common


#: gt4py builtin and operator names whose Array API name differs.
STANDARD_NAMES: Final[Mapping[str, str]] = {
    "neg": "negative",
    "pos": "positive",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "floordiv": "floor_divide",
    "power": "pow",
    "mod": "remainder",
    "invert": "logical_not",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
}


# TODO(havogt): Redesign coherent conversion rules when arrays/scalars of different array
#   libraries are used together.
@dataclasses.dataclass(frozen=True)
class ArrayOps:
    """
    Operations on the buffers of embedded fields for one array namespace.

    `xp` is the Array API namespace of the buffers. Field code uses `xp` directly
    for standard functions and the methods of this class wherever a library may
    need to differ; those methods are the override points.
    """

    #: Replacements of whole `fbuiltins` implementations for fields of this namespace.
    builtin_overrides: ClassVar[Mapping[Any, Callable[..., Any]]] = {}

    xp: ModuleType
    _elementwise_function_cache: dict[str, Callable[..., Any]] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.xp.__name__})"

    @property
    def array_ns(self) -> ModuleType:
        """The library's own module, as exposed by `NdArrayField.array_ns`."""
        return self.xp

    def dtype(self, scalar_type: type) -> Any:
        return core_ndarray_utils.namespace_dtype(self.xp, core_defs.dtype(scalar_type).scalar_type)

    def scalar_type(self, dtype: Any) -> type[core_defs.Scalar]:
        return core_ndarray_utils.scalar_type(self.xp, dtype)

    def device(self, array: core_defs.NDArrayObject) -> Any:
        return array_api_compat.device(array)

    def asarray(
        self,
        data: Any,
        *,
        dtype: Any = None,
        like: core_defs.NDArrayObject | None = None,
    ) -> core_defs.NDArrayObject:
        """`data` as an array of this namespace, on the device of `like` if given."""
        return self.xp.asarray(
            data,
            dtype=None if dtype is None else self.dtype(dtype),
            device=None if like is None else self.device(like),
        )

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        """
        `value` as an operand of an elementwise function next to the array `like`.

        Python scalars stay Python scalars, so they keep the weak typing of the
        standard. NumPy scalars become 0-d arrays of their dtype, which promotes like
        NumPy does with NumPy scalars. They reach fields of every library, since gt4py's
        scalar types are NumPy's, but the standard does not accept them as operands.
        Libraries that do (NumPy, CuPy, JAX) override this to keep them as they are,
        which spares creating a 0-d array, on the device, per operation.
        """
        if isinstance(value, np.generic):
            return self.asarray(value, dtype=type(value), like=like)
        return value

    def elementwise_function(self, name: str) -> Callable[..., Any]:
        """The elementwise function for a gt4py builtin or operator name."""
        try:
            return self._elementwise_function_cache[name]
        except KeyError:
            return self._elementwise_function_cache.setdefault(
                name, self._elementwise_function(name)
            )

    def _elementwise_function(self, name: str) -> Callable[..., Any]:
        if name == "where":
            return self.where
        func = getattr(self.xp, STANDARD_NAMES.get(name, name), None)
        if func is None:
            raise NotImplementedError(
                f"'{name}' is not implemented for array namespace '{self.xp.__name__}'."
            )
        return func

    def where(self, condition: Any, x: Any, y: Any) -> core_defs.NDArrayObject:
        """
        `xp.where` for the operands of the `where` builtin.

        The standard requires an array `condition`, but the builtin also receives a
        scalar one (e.g. from `broadcast(True)`), which some libraries reject (CuPy,
        torch). It becomes a 0-d array on the device of the other operands. `x` and `y`
        are converted like the operands of any other elementwise function; the standard
        also requires one of them to be an array, so if both are scalars, `x` becomes a
        0-d array with the default dtype for its kind, as NumPy promotes it.
        """
        if isinstance(condition, bool | np.bool_):
            like = next(
                a
                for a in (x, y)
                if array_api_compat.is_array_api_obj(a) and not isinstance(a, np.generic)
            )
            condition = self.asarray(condition, dtype=bool, like=like)
        x, y = self.operand(x, condition), self.operand(y, condition)
        if not (array_api_compat.is_array_api_obj(x) or array_api_compat.is_array_api_obj(y)):
            x = self.asarray(x, like=condition)
        return self.xp.where(condition, x, y)

    def setitem(
        self, array: core_defs.NDArrayObject, key: common.RelativeIndexSequence, value: Any
    ) -> core_defs.NDArrayObject:
        """
        Write `value` into `array[key]`; returns the array holding the result.

        `value` is a Python scalar, a NumPy scalar (gt4py's scalar types are NumPy's)
        or an array of the same library as `array`; arrays of other libraries raise
        `TypeError`. The standard leaves assigning a value of another dtype unspecified,
        so it is cast to the dtype of `array` first. Libraries that cast on assignment
        (NumPy, CuPy) assign directly; JAX arrays are immutable and need
        `.at[key].set(value)`.
        """
        if isinstance(value, np.generic):
            value = self.xp.asarray(value, device=self.device(array))
        elif not isinstance(value, bool | int | float):
            if array_api_compat.array_namespace(value) is not self.xp:
                raise TypeError(
                    f"Cannot assign an array of '{type(value).__module__}' to an array of"
                    f" '{self.xp.__name__}'."
                )
        if not isinstance(value, bool | int | float) and value.dtype != array.dtype:
            value = self.xp.astype(value, array.dtype)
        array[key] = value  # type: ignore[index] # `NDArrayObject` typing is not complete
        return array

    # The operations on a connectivity table. `static` is what the connectivity holds for the
    # library of its table: `None`, unless the library created the connectivity with one.

    def hyperslice(
        self,
        table: core_defs.NDArrayObject,
        image_range: common.UnitRange,
        skip_value: Optional[core_defs.IntegralScalar] = None,
        *,
        static: Any = None,
    ) -> Optional[tuple[slice, ...]]:
        """
        Return the hypercube slice that contains all indices in `table` that are within `image_range`, or `None` if no such hypercube exists.

        If `skip_value` is given, the selected values are ignored. It returns the smallest hypercube.
        A bigger hypercube could be constructed by adding lines that contain only `skip_value`s.

        Example:
            table =  0  1 -1
                     3  4 -1
                    -1 -1 -1
            skip_value = -1

            would currently select the 2x2 range [0,2], [0,2], but could also select the 3x3 range [0,3], [0,3].

        A library which cannot read the values of `table` returns all of it. What an entry
        outside of `image_range` reads is then defined by its `gather`.
        """
        xp = self.xp
        select_mask = (table >= image_range.start) & (table < image_range.stop)

        if not xp.any(select_mask):
            return None

        nnz: tuple[core_defs.NDArrayObject, ...] = xp.nonzero(select_mask)

        slices = tuple(
            slice(int(xp.min(dim_nnz_indices)), int(xp.max(dim_nnz_indices)) + 1)
            for dim_nnz_indices in nnz
        )
        hcube = select_mask[tuple(slices)]
        if skip_value is not None:
            ignore_mask = table == skip_value
            hcube = hcube | ignore_mask[tuple(slices)]
        if not xp.all(hcube):
            return None

        return slices

    def restrict_table(
        self,
        table: core_defs.NDArrayObject,
        key: common.RelativeIndexSequence,
        *,
        static: Any = None,
    ) -> core_defs.NDArrayObject:
        """Return `table[key]`."""
        return self.asarray(table[key])

    def gather(
        self, array: core_defs.NDArrayObject, indices: tuple[core_defs.NDArrayObject, ...]
    ) -> core_defs.NDArrayObject:
        """Return `array[indices]` for one integer index array per axis, all of the same shape."""
        return array[indices]

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return np.from_dlpack(array, device="cpu", copy=None)  # type: ignore[arg-type] # `NDArrayObject` typing is not complete

    def buffer_info(self, array: core_defs.NDArrayObject) -> common.BufferInfo:
        return common.BufferInfo.from_ndarray(array)


_ops_by_namespace: dict[ModuleType, ArrayOps] = {}
_ops_by_array_type: dict[type, ArrayOps] = {}


def register(ops: ArrayOps) -> None:
    """Use `ops` for the buffers whose compat namespace is `ops.xp`, replacing an earlier registration."""
    _ops_by_namespace[ops.xp] = ops
    _ops_by_array_type.clear()


def for_namespace(xp: ModuleType) -> ArrayOps:
    """The registered `ArrayOps` for the compat namespace `xp`, or the default one."""
    try:
        return _ops_by_namespace[xp]
    except KeyError:
        ops = _ops_by_namespace[xp] = ArrayOps(xp)
        return ops


def of(array: Any) -> ArrayOps:
    """The `ArrayOps` for the namespace of `array`; raises `TypeError` if it is not an array."""
    try:
        return _ops_by_array_type[type(array)]
    except KeyError:
        ops = _ops_by_array_type[type(array)] = for_namespace(
            array_api_compat.array_namespace(array)
        )
        return ops


def default() -> ArrayOps:
    """The `ArrayOps` used for data that is not an array, e.g. nested lists."""
    return for_namespace(compat_numpy)
