# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The embedded implementation of the builtins declared in `ffront.fbuiltins` and `ffront.experimental`.

Calling a builtin with values is embedded execution. In this order, a call:

1. maps over the tuples and named collections the builtin accepts (`where`, `concat_where`,
   `astype`);
2. runs the implementation of the first argument whose type implements the builtin
   (`__gt_builtin_func__`, e.g. a field);
3. runs the implementation for scalars.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any, Final, NoReturn, cast

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import named_collections
from gt4py.next.ffront import experimental, fbuiltins


# `fbuiltins.astype` is declared with overloads, which mypy sees instead of the `BuiltInFunction`
_ASTYPE: Final = cast(fbuiltins.BuiltInFunction, fbuiltins.astype)


def _no_implementation(builtin: fbuiltins.BuiltInFunction, args: tuple[Any, ...]) -> NoReturn:
    raise TypeError(
        f"'{builtin.name}' has no embedded implementation for arguments of type"
        f" ({', '.join(type(arg).__name__ for arg in args)})."
    )


def _call(builtin: fbuiltins.BuiltInFunction, /, *args: Any, **kwargs: Any) -> Any:
    if (map_tree := _TREE_MAPS.get(builtin)) is not None and (
        result := map_tree(builtin, *args)
    ) is not NotImplemented:
        return result
    for arg in args:
        if (dispatcher := getattr(type(arg), "__gt_builtin_func__", None)) is not None and (
            implementation := dispatcher(builtin)
        ) is not NotImplemented:
            return implementation(*args, **kwargs)
    if (implementation := _SCALAR_IMPLEMENTATIONS.get(builtin)) is not None:
        return implementation(*args, **kwargs)
    _no_implementation(builtin, args)


# -- tuples and named collections --


def _map_branches(
    builtin: fbuiltins.BuiltInFunction, cond: Any, true_field: Any, false_field: Any
) -> Any:
    if isinstance(true_field, named_collections.CUSTOM_NAMED_COLLECTION_TYPES) or isinstance(
        false_field, named_collections.CUSTOM_NAMED_COLLECTION_TYPES
    ):
        if type(true_field) is not type(false_field):
            raise ValueError(
                f"Either both or none can be a named collection of the same type in '{true_field=}' and '{false_field=}'."
            )
        return named_collections.tree_map_named_collection(lambda t, f: builtin(cond, t, f))(
            true_field, false_field
        )
    if isinstance(true_field, tuple) or isinstance(false_field, tuple):
        if not (isinstance(true_field, tuple) and isinstance(false_field, tuple)):
            raise ValueError(
                # TODO(havogt) find a strategy to unify parsing and embedded error messages
                f"Either both or none can be tuple in '{true_field=}' and '{false_field=}'."
            )
        if len(true_field) != len(false_field):
            raise ValueError(
                "Tuple of different size not allowed."
            )  # TODO(havogt) find a strategy to unify parsing and embedded error messages
        return tuple(builtin(cond, t, f) for t, f in zip(true_field, false_field))
    return NotImplemented


def _map_value(builtin: fbuiltins.BuiltInFunction, value: Any, type_: type) -> Any:
    if isinstance(value, tuple):
        return tuple(builtin(v, type_) for v in value)
    return NotImplemented


_TREE_MAPS: Final[dict[fbuiltins.BuiltInFunction, Callable[..., Any]]] = {
    fbuiltins.where: _map_branches,
    experimental.concat_where: _map_branches,
    _ASTYPE: _map_value,
}


# -- scalars --


def _unary(
    builtin: fbuiltins.BuiltInFunction, function: Callable[[Any], Any]
) -> Callable[..., Any]:
    """The implementation of `builtin` for a scalar: `function`."""

    def implementation(value: core_defs.Scalar, /) -> Any:
        if not core_defs.is_scalar_type(value):
            _no_implementation(builtin, (value,))
        return function(value)

    return implementation


def _binary(
    builtin: fbuiltins.BuiltInFunction, function: Callable[[Any, Any], Any]
) -> Callable[..., Any]:
    """The implementation of `builtin` for two scalars: `function`."""

    def implementation(lhs: core_defs.Scalar, rhs: core_defs.Scalar, /) -> Any:
        if not (core_defs.is_scalar_type(lhs) and core_defs.is_scalar_type(rhs)):
            _no_implementation(builtin, (lhs, rhs))
        return function(lhs, rhs)

    return implementation


def _broadcast(field: core_defs.Scalar, dims: tuple[Any, ...], /) -> core_defs.Scalar:
    if not core_defs.is_scalar_type(field):
        _no_implementation(fbuiltins.broadcast, (field, dims))
    # TODO(havogt) implement with FunctionField, the workaround is to ignore broadcasting on scalars as they broadcast automatically, but we lose the check for compatible dimensions
    return field


def _astype(value: core_defs.Scalar, type_: type, /) -> core_defs.Scalar:
    if not core_defs.is_scalar_type(value):
        _no_implementation(_ASTYPE, (value, type_))
    return core_defs.dtype(type_).scalar_type(value)


try:
    from scipy.special import gamma as _gamma  # dtype-preserving ufunc
except ImportError:

    def _gamma(value: core_defs.ScalarT) -> core_defs.ScalarT:
        # restore the input scalar type, which `math.gamma` widens to `float`
        return type(value)(math.gamma(value))


_UNARY_MATH_FUNCTIONS: Final[dict[str, Callable[[Any], Any]]] = {
    "abs": abs,
    "neg": operator.neg,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "gamma": _gamma,
    "cbrt": np.cbrt,
    "floor": np.floor,
    "ceil": np.ceil,
    "trunc": np.trunc,
    "isfinite": np.isfinite,
    "isinf": np.isinf,
    "isnan": np.isnan,
}

_SCALAR_IMPLEMENTATIONS: Final[dict[fbuiltins.BuiltInFunction, Callable[..., Any]]] = {
    fbuiltins.broadcast: _broadcast,
    _ASTYPE: _astype,
    **{
        getattr(fbuiltins, name): _unary(getattr(fbuiltins, name), function)
        for name, function in _UNARY_MATH_FUNCTIONS.items()
    },
    **{
        getattr(fbuiltins, name): _binary(getattr(fbuiltins, name), getattr(np, name))
        for name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES
    },
}

#: The builtins that need a field argument: called with scalars only, they raise
_FIELD_ONLY_BUILTINS: Final = frozenset(
    {
        fbuiltins.neighbor_sum,
        fbuiltins.max_over,
        fbuiltins.min_over,
        fbuiltins.where,
        experimental.as_offset,
        experimental.concat_where,
    }
)

ALL_BUILTINS: Final = frozenset(
    {getattr(fbuiltins, name) for name in fbuiltins.FUN_BUILTIN_NAMES}
    | {getattr(experimental, name) for name in experimental.EXPERIMENTAL_FUN_BUILTIN_NAMES}
)

assert (
    _SCALAR_IMPLEMENTATIONS.keys() | _FIELD_ONLY_BUILTINS == ALL_BUILTINS
    and not _SCALAR_IMPLEMENTATIONS.keys() & _FIELD_ONLY_BUILTINS
), "Every builtin needs an embedded implementation for scalars, or to be field-only."

fbuiltins.call_implementation = _call
