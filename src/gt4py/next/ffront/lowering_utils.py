# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from typing import Callable, Optional, TypeVar

from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_specifications as ts


# TODO(tehrengruber): The code quality of this function is poor. We should rewrite it.
def process_elements(
    process_func: Callable[..., itir.Expr],
    objs: itir.Expr | Iterable[itir.Expr],
    current_el_type: ts.TypeSpec,
    arg_types: Optional[Iterable[ts.TypeSpec]] = None,
) -> itir.FunCall:
    """
    Recursively applies a processing function to all primitive constituents of a tuple.

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

    let_ids = tuple(f"__val_{eve_utils.content_hash(obj)}" for obj in objs)
    body = _process_elements_impl(
        process_func,
        tuple(im.ref(let_id) for let_id in let_ids),
        current_el_type,
        arg_types=arg_types,
    )

    return im.let(*(zip(let_ids, objs, strict=True)))(body)


T = TypeVar("T", bound=itir.Expr, covariant=True)


def _process_elements_impl(
    process_func: Callable[..., itir.Expr],
    _current_el_exprs: Iterable[T],
    current_el_type: ts.TypeSpec,
    arg_types: Optional[Iterable[ts.TypeSpec]],
) -> itir.Expr:
    if isinstance(current_el_type, ts.TupleType):
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
