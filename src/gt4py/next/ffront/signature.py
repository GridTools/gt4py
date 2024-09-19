# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

# TODO(ricoh): This overlaps with `canonicalize_arguments`, solutions:
# - merge the two
# - extract the signature gathering functionality from canonicalize_arguments
#   and use it to pass the signature through the toolchain so that the
#   decorate step can take care of it. Then get rid of all pre-toolchain
#   arguments rearranging (including this module)

from __future__ import annotations

import functools
import inspect
import types
from typing import Any, Callable

from gt4py.next.ffront import (
    field_operator_ast as foast,
    program_ast as past,
    stages as ffront_stages,
)
from gt4py.next.type_system import type_specifications as ts


def should_be_positional(param: inspect.Parameter) -> bool:
    return (param.kind is inspect.Parameter.POSITIONAL_ONLY) or (
        param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


@functools.singledispatch
def make_signature(func: Any) -> inspect.Signature:
    """Make a signature for a Python or DSL callable, which suffices for use in 'convert_to_positional'."""
    if isinstance(func, types.FunctionType):
        return inspect.signature(func)
    raise NotImplementedError(f"'make_signature' not implemented for {type(func)}.")


@make_signature.register(foast.ScanOperator)
@make_signature.register(past.Program)
@make_signature.register(foast.FieldOperator)
def signature_from_fieldop(func: foast.FieldOperator) -> inspect.Signature:
    if isinstance(func.type, ts.DeferredType):
        raise NotImplementedError(
            f"'make_signature' not implemented for pre type deduction {type(func)}."
        )
    fieldview_signature = func.type.definition
    return inspect.Signature(
        parameters=[
            inspect.Parameter(name=str(i), kind=inspect.Parameter.POSITIONAL_ONLY)
            for i, param in enumerate(fieldview_signature.pos_only_args)
        ]
        + [
            inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for k in fieldview_signature.pos_or_kw_args
        ],
    )


@make_signature.register(ffront_stages.FieldOperatorDefinition)
def signature_from_fieldop_def(func: ffront_stages.FieldOperatorDefinition) -> inspect.Signature:
    signature = make_signature(func.definition)
    if func.node_class == foast.ScanOperator:
        return inspect.Signature(list(signature.parameters.values())[1:])
    return signature


@make_signature.register(ffront_stages.ProgramDefinition)
def signature_from_program_def(func: ffront_stages.ProgramDefinition) -> inspect.Signature:
    return make_signature(func.definition)


@make_signature.register(ffront_stages.FoastOperatorDefinition)
def signature_from_foast_stage(func: ffront_stages.FoastOperatorDefinition) -> inspect.Signature:
    return make_signature(func.foast_node)


@make_signature.register
def signature_from_past_stage(func: ffront_stages.PastProgramDefinition) -> inspect.Signature:
    return make_signature(func.past_node)


def convert_to_positional(
    func: Callable
    | foast.FieldOperator
    | foast.ScanOperator
    | ffront_stages.FieldOperatorDefinition
    | ffront_stages.FoastOperatorDefinition
    | ffront_stages.ProgramDefinition
    | ffront_stages.PastProgramDefinition,
    *args: Any,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Convert arguments given as keyword args to positional ones where possible.

    Raises en error if and only if there are clearly missing positional arguments,
    Without awareness of the peculiarities of DSL function signatures. A more
    thorough check on whether the signature is fulfilled is expected to happen
    later in the toolchain.

    Note that positional-or-keyword arguments with defaults will have their defaults
    inserted even if not strictly necessary. This is to reduce complexity and should
    be changed if the current behavior is found harmful in some way.

    Examples:
    >>> def example(posonly, /, pos_or_key, pk_with_default=42, *, key_only=43):
    ...     pass

    >>> convert_to_positional(example, 1, pos_or_key=2, key_only=3)
    ((1, 2, 42), {'key_only': 3})
    >>> # inserting the default value '42' here could be avoided
    >>> # but this is not the current behavior.
    """
    signature = make_signature(func)
    new_args = list(args)
    modified_kwargs = kwargs.copy()
    missing = []
    interesting_params = [p for p in signature.parameters.values() if should_be_positional(p)]

    for param in interesting_params[len(args) :]:
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and param.name in modified_kwargs:
            # if keyword allowed, check if was given as kwarg
            new_args.append(modified_kwargs.pop(param.name))
        else:
            # add default and report as missing if no default
            # note: this treats POSITIONAL_ONLY params correctly, as they can not have a default.
            new_args.append(param.default)
            if param.default is inspect._empty:
                missing.append(param.name)
    if missing:
        raise TypeError(f"Missing positional argument(s): {', '.join(missing)}.")
    return tuple(new_args), modified_kwargs
