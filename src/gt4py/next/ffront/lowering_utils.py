# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from collections.abc import Iterable
from typing import Callable, Optional, TypeVar

from gt4py.next import utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.type_system import type_info, type_specifications as ts


# TODO(tehrengruber): The code quality of this function is poor. We should rewrite it.
def process_elements(
    process_func: Callable[..., itir.Expr],
    objs: itir.Expr | Iterable[itir.Expr],
    current_el_type: ts.TypeSpec,
    arg_types: Optional[Iterable[ts.TypeSpec]] = None,
) -> itir.FunCall:
    """
    Recursively applies a processing function to all primitive constituents of a tuple or
    named collection.

    Arguments:
        process_func: A callable that takes an itir.Expr representing a leaf-element of `objs`.
            If multiple `objs` are given the callable takes equally many arguments.
        objs: The object whose elements are to be transformed.
        current_el_type: A type with the same structure as the elements of `objs`. The leaf-types
            are not used and thus not relevant.
        arg_types: If provided, a tuple of the type of each argument is passed to `process_func` as last argument.
            Note, that `arg_types` might coincide with `(current_el_type,)*len(objs)`, but not necessarily,
            in case of implicit broadcasts.
    """
    if isinstance(objs, itir.Expr):
        objs = (objs,)

    var_names = tuple(f"__val_{obj.fingerprint()}" for obj in objs)
    # Note: The same `var_name` might appear multiple times if the same object appears multiple
    # times. Since we use a dict to collect the bound variables, this is fine.
    bound_vars = {var_name: obj for var_name, obj in zip(var_names, objs)}
    body = _process_elements_impl(
        process_func,
        tuple(im.ref(var_name) for var_name in var_names),
        current_el_type,
        arg_types=arg_types,
    )

    return im.let(*bound_vars.items())(body)


T = TypeVar("T", bound=itir.Expr, covariant=True)


def _process_elements_impl(
    process_func: Callable[..., itir.Expr],
    _current_el_exprs: Iterable[T],
    current_el_type: ts.TypeSpec,
    arg_types: Optional[Iterable[ts.TypeSpec]],
) -> itir.Expr:
    if isinstance(current_el_type, (ts.TupleType, ts.NamedCollectionType)):
        result = im.make_tuple(
            *(
                _process_elements_impl(
                    process_func,
                    tuple(
                        im.tuple_get(i, current_el_expr) for current_el_expr in _current_el_exprs
                    ),
                    current_el_type.types[i],
                    arg_types=tuple(arg_t.types[i] for arg_t in arg_types)  # type: ignore[attr-defined] # guaranteed by the requirement that `current_el_type` and each element of `arg_types` have the same tuple structure
                    if arg_types is not None
                    else None,
                )
                for i in range(len(current_el_type.types))
            )
        )
    else:
        if arg_types is not None:
            result = process_func(*_current_el_exprs, arg_types)
        else:
            result = process_func(*_current_el_exprs)

    return result


def _collapsing_tuple_get(expr: itir.Expr, i: int) -> itir.Expr:
    """Like `im.tuple_get`, but collapses immediately when `expr` is a `make_tuple` call.

    Note: argument order is `(expr, i)` to allow use as a `functools.reduce` reducer.
    """
    if cpm.is_call_to(expr, "make_tuple"):
        return expr.args[i]
    return im.tuple_get(i, expr)


def _tree_map_tuple_body(
    f: itir.Expr, tup_exprs: list[itir.Expr], tup_types: list[ts.TupleType]
) -> itir.Expr:
    """Recursively unroll `tree_map_tuple(f)(t1, ..., tN)` into `make_tuple` calls."""

    @utils.tree_map(
        collection_type=ts.TupleType,
        result_collection_constructor=lambda _, elts: im.make_tuple(*elts),
        with_path_arg=True,
    )
    def mapper(*args: ts.TypeSpec | tuple[int, ...]) -> itir.Expr:
        *_el_types, path = args
        assert isinstance(path, tuple), "Expected path to be tuple[int, ...]"
        return im.call(f)(
            *(functools.reduce(_collapsing_tuple_get, path, tup_expr) for tup_expr in tup_exprs)
        )

    return mapper(*tup_types)


def _map_tuple_body(
    f: itir.Expr, tup_exprs: list[itir.Expr], tup_types: list[ts.TupleType]
) -> itir.Expr:
    """Unroll `map_tuple(f)(t)` over top-level elements only (no recursion)."""
    (tup_expr,) = tup_exprs
    (tup_type,) = tup_types
    return im.make_tuple(
        *(im.call(f)(_collapsing_tuple_get(tup_expr, i)) for i in range(len(tup_type.types)))
    )


_UNROLLERS = {
    "tree_map_tuple": _tree_map_tuple_body,
    "map_tuple": _map_tuple_body,
}


def _unroll_tuple_map(
    builtin_name: str,
    f: itir.Expr,
    tup_exprs: Iterable[itir.Expr],
    tup_types: Iterable[ts.TypeSpec],
    *,
    uids: utils.IDGeneratorPool,
) -> itir.Expr:
    tup_exprs = list(tup_exprs)
    tup_types_list = list(tup_types)
    for tup_type in tup_types_list:
        if not isinstance(tup_type, ts.TupleType):
            raise TypeError(
                f"'{builtin_name}' requires all arguments to be tuples, got '{tup_type}'."
            )
    tup_types_validated: list[ts.TupleType] = tup_types_list  # type: ignore[assignment]

    if not type_info.tuple_structures_match(*tup_types_validated):
        raise TypeError(
            f"'{builtin_name}' requires all arguments to share the same (nested) tuple "
            f"structure, got {[str(t) for t in tup_types_validated]}."
        )

    # For trivial args (those that can be duplicated without cost or side effects),
    # we substitute them directly into the body. This avoids leaving behind
    # `tuple_get(i, make_tuple(...))` patterns that would otherwise require a
    # separate cleanup pass (`CollapseTuple`). For non-trivial args we still
    # introduce a `let` binding to avoid duplicating expensive sub-expressions.
    substituted_exprs: list[itir.Expr] = []
    let_bindings: list[tuple[str, itir.Expr]] = []
    for tup in tup_exprs:
        if isinstance(tup, (itir.SymRef, itir.Literal)) or cpm.is_call_to(tup, "make_tuple"):
            substituted_exprs.append(tup)
        else:
            ref_name = next(uids["__utm"])
            let_bindings.append((ref_name, tup))
            substituted_exprs.append(im.ref(ref_name))

    body = _UNROLLERS[builtin_name](f, substituted_exprs, tup_types_validated)
    return im.let(*let_bindings)(body) if let_bindings else body


def unroll_tree_map_tuple(
    f: itir.Expr,
    tup_exprs: Iterable[itir.Expr],
    tup_types: Iterable[ts.TypeSpec],
    *,
    uids: utils.IDGeneratorPool,
) -> itir.Expr:
    """
    Lower ``tree_map_tuple(f)(t1, ..., tN)`` to explicit ``make_tuple`` calls, recursing into
    nested tuples and applying ``f`` to each leaf.

    Args:
        f: The function to apply at each leaf.
        tup_exprs: The (already lowered) tuple argument expressions.
        tup_types: The type of each argument in ``tup_exprs``; all must be ``TupleType`` and
            share the same (nested) structure.
        uids: Used to generate fresh names for `let`-bindings of non-trivial arguments.
    """
    return _unroll_tuple_map("tree_map_tuple", f, tup_exprs, tup_types, uids=uids)


def unroll_map_tuple(
    f: itir.Expr,
    tup_expr: itir.Expr,
    tup_type: ts.TypeSpec,
    *,
    uids: utils.IDGeneratorPool,
) -> itir.Expr:
    """
    Lower ``map_tuple(f)(t)`` to an explicit ``make_tuple`` call, applying ``f`` to each
    top-level element only (no recursion).

    Args:
        f: The function to apply to each top-level element.
        tup_expr: The (already lowered) tuple argument expression.
        tup_type: The type of ``tup_expr``; must be a ``TupleType``.
        uids: Used to generate a fresh name for a `let`-binding of a non-trivial argument.
    """
    return _unroll_tuple_map("map_tuple", f, (tup_expr,), (tup_type,), uids=uids)
