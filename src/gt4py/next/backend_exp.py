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

from __future__ import annotations

import dataclasses
import functools
import inspect
import types
import typing
from typing import Any, Callable, Generic, Optional

from gt4py.next import backend
from gt4py.next.ffront import (
    field_operator_ast as foast,
    foast_to_itir,
    foast_to_past,
    func_to_foast,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
    type_specifications as ffts,
)
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, workflow
from gt4py.next.type_system import type_specifications as ts


DataT = typing.TypeVar("DataT")


@dataclasses.dataclass(frozen=True)
class IteratorProgramMetadata:
    fieldview_program_type: ffts.ProgramType


@dataclasses.dataclass(frozen=True)
class DataWithIteratorProgramMetadata(Generic[DataT]):
    data: DataT
    metadata: IteratorProgramMetadata


@dataclasses.dataclass(frozen=True)
class ItirShim:
    definition: ffront_stages.FoastOperatorDefinition
    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr]

    def __gt_closure_vars__(self) -> Optional[dict[str, Any]]:
        return self.definition.closure_vars

    def __gt_type__(self) -> ts.CallableType:
        return self.definition.foast_node.type

    def __gt_itir__(self) -> itir.FunctionDefinition:
        return self.foast_to_itir(self.definition)


def should_be_positional(param: inspect.Parameter) -> bool:
    return (param.kind is inspect.Parameter.POSITIONAL_ONLY) or (
        param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )


@functools.singledispatch
def make_signature(func: Any) -> inspect.Signature:
    """Make a signature for a Python or DSL callable, which suffices for use in 'convert_to_positional'."""
    raise NotImplementedError(f"'make_signature' not implemented for {type(func)}.")


@make_signature.dispatch
def signature_from_callable(func: types.FunctionType) -> inspect.Signature:
    return inspect.signature(func)


@make_signature.dispatch(foast.FieldOperator)
@make_signature.dispatch(foast.ScanOperator)
def signature_from_fieldop(func: foast.FieldOperator | foast.ScanOperator) -> inspect.Signature:
    fieldview_signature = func.type.definition
    return inspect.Signature(
        parameters={
            **{
                i: inspect.Parameter(name=i, kind=inspect.Parameter.POSITIONAL_ONLY)
                for i, param in enumerate(fieldview_signature.pos_only_args)
            },
            **{
                k: inspect.Parameter(name=k, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for k in fieldview_signature.pos_or_kw_args
            },
        }
    )


def convert_to_positional(
    func: Callable | foast.FieldOperator | foast.ScanOperator, *args: Any, **kwargs: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Convert arguments given as keyword args to positional ones where possible.

    Raises en error if and only if there are clearly missing positional arguments,
    Without awareness of the peculiarities of DSL function signatures. A more
    thorough check on whether the signature is fulfilled is expected to happen
    later in the toolchain.

    Examples:
    >>> def example(posonly, /, pos_or_key, pk_with_default=42, *, key_only=43):
    ...     pass
    >>> convert_to_positional(example, 1, pos_or_key=2, key_only=3)
    ((1, 2), {"key_only": 3})
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


@dataclasses.dataclass(frozen=True)
class FieldviewOpToFieldviewProg:
    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr] = (
        dataclasses.field(
            default_factory=lambda: workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )
    fop_args_to_prog_args: workflow.Workflow[
        ffront_stages.PastClosure, ffront_stages.PastClosure
    ] = dataclasses.field(default_factory=lambda: past_process_args.PastProcessArgs(aot_off=True))

    def __call__(
        self,
        inp: workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.CompileArgSpec],
    ) -> workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec]:
        fieldview_program = foast_to_past.foast_to_past(
            ffront_stages.FoastWithTypes(
                foast_op_def=inp.data,
                arg_types=tuple(arg.gt_type for arg in inp.args.args),
                kwarg_types={
                    k: v.gt_type for k, v in inp.args.kwargs.items() if k not in ["from_fieldop"]
                },
                closure_vars={inp.data.foast_node.id: ItirShim(inp.data, self.foast_to_itir)},
            )
        )
        transformed = self.fop_args_to_prog_args(
            ffront_stages.PastClosure(
                definition=fieldview_program, args=inp.args.args, kwargs=inp.args.kwargs
            )
        )
        return workflow.DataWithArgs(
            data=fieldview_program,
            args=dataclasses.replace(inp.args, args=transformed.args, kwargs=transformed.kwargs),
        )


@dataclasses.dataclass(frozen=True)
class PastToItirAdapter:
    step: workflow.Workflow[ffront_stages.AOTFieldviewProgramAst, stages.AOTProgram] = (
        dataclasses.field(default_factory=past_to_itir.PastToItir)
    )

    def __call__(
        self, inp: workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec]
    ) -> stages.AOTProgram:
        aot_fvprog = ffront_stages.AOTFieldviewProgramAst(definition=inp.data, argspec=inp.args)
        return self.step(aot_fvprog)


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_fieldview_op: workflow.Workflow[
        workflow.DataWithArgs[
            ffront_stages.FieldOperatorDefinition | ffront_stages.FoastOperatorDefinition,
            stages.CompileArgSpec,
        ],
        workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.CompileArgSpec],
    ] = dataclasses.field(
        default_factory=lambda: workflow.DataOnlyAdapter(
            func_to_foast.OptionalFuncToFoastFactory(cached=True)
        )
    )
    field_view_op_to_prog: workflow.Workflow[
        workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.CompileArgSpec],
        workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec],
    ] = dataclasses.field(default_factory=FieldviewOpToFieldviewProg)

    past_to_itir: workflow.Workflow[
        workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec],
        stages.AOTProgram,
    ] = dataclasses.field(default_factory=PastToItirAdapter)


class ExpBackend(backend.Backend):
    def __call__(
        self,
        program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(
            program, (ffront_stages.FieldOperatorDefinition, ffront_stages.FoastOperatorDefinition)
        ):
            _ = kwargs.pop("from_fieldop")
            # taking the offset provider out is not needed
            program_info = self.transforms_fop(
                workflow.DataWithArgs(
                    data=program,
                    args=stages.CompileArgSpec.from_concrete_no_size(*args, **kwargs),
                )
            )
            self.executor(program_info, *args, **kwargs)
        else:
            super().__call__(program, *args, **kwargs)
