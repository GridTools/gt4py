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
    func_to_past,
    past_process_args,
    past_to_itir,
    program_ast as past,
    stages as ffront_stages,
    type_specifications as ffts,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, workflow
from gt4py.next.type_system import type_specifications as ts, type_translation


DataT = typing.TypeVar("DataT")


ARGS: typing.TypeAlias = stages.JITArgs
CARG: typing.TypeAlias = stages.CompileArgSpec
DSL_FOP: typing.TypeAlias = ffront_stages.FieldOperatorDefinition
FOP: typing.TypeAlias = ffront_stages.FoastOperatorDefinition
DSL_PRG: typing.TypeAlias = ffront_stages.ProgramDefinition
PRG: typing.TypeAlias = ffront_stages.PastProgramDefinition
IT_PRG: typing.TypeAlias = itir.FencilDefinition


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

    def __gt_itir__(self) -> itir.Expr:
        return self.foast_to_itir(self.definition)


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
    func: Callable | foast.FieldOperator | foast.ScanOperator | DSL_FOP | FOP | DSL_PRG | PRG,
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


@dataclasses.dataclass(frozen=True)
class FieldviewOpToFieldviewProg:
    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr] = (
        dataclasses.field(
            default_factory=lambda: workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )

    def __call__(
        self,
        inp: workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.CompileArgSpec],
    ) -> workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec]:
        fieldview_program = foast_to_past.foast_to_past(
            ffront_stages.FoastWithTypes(
                foast_op_def=inp.data,
                arg_types=tuple(type_translation.from_value(arg) for arg in inp.args.args),
                kwarg_types={
                    k: type_translation.from_value(v)
                    for k, v in inp.args.kwargs.items()
                    if k not in ["from_fieldop"]
                },
                closure_vars={inp.data.foast_node.id: ItirShim(inp.data, self.foast_to_itir)},
            )
        )
        return workflow.DataWithArgs(
            data=fieldview_program,
            args=inp.args,
        )


def transform_prog_args(
    inp: workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec],
) -> workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec]:
    transformed = past_process_args.PastProcessArgs(aot_off=True)(
        ffront_stages.PastClosure(definition=inp.data, args=inp.args.args, kwargs=inp.args.kwargs)
    )
    return workflow.DataWithArgs(
        data=transformed.definition,
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


def jit_to_aot_args(inp: stages.JITArgs) -> stages.CompileArgSpec:
    return stages.CompileArgSpec.from_concrete_no_size(*inp.args, **inp.kwargs)


AOT_FOP: typing.TypeAlias = workflow.DataWithArgs[FOP, CARG]
AOT_PRG: typing.TypeAlias = workflow.DataWithArgs[PRG, CARG]


INPUT_DATA_T: typing.TypeAlias = DSL_FOP | FOP | DSL_PRG | PRG | IT_PRG


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.MultiWorkflow):
    """Modular workflow for transformations with access to intermediates."""

    aotify_args: workflow.Workflow[
        workflow.DataWithArgs[INPUT_DATA_T, ARGS], workflow.DataWithArgs[INPUT_DATA_T, CARG]
    ] = dataclasses.field(default_factory=lambda: workflow.ArgsOnlyAdapter(jit_to_aot_args))

    func_to_foast: workflow.Workflow[workflow.DataWithArgs[DSL_FOP | FOP, CARG], AOT_FOP] = (
        dataclasses.field(
            default_factory=lambda: workflow.DataOnlyAdapter(
                func_to_foast.OptionalFuncToFoastFactory(cached=True)
            )
        )
    )

    func_to_past: workflow.Workflow[workflow.DataWithArgs[DSL_PRG | PRG, CARG], AOT_PRG] = (
        dataclasses.field(
            default_factory=lambda: workflow.DataOnlyAdapter(
                func_to_past.OptionalFuncToPastFactory(cached=True)
            )
        )
    )

    foast_to_itir: workflow.Workflow[AOT_FOP, itir.Expr] = dataclasses.field(
        default_factory=lambda: workflow.StripArgsAdapter(
            workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )

    field_view_op_to_prog: workflow.Workflow[workflow.DataWithArgs[FOP, CARG], AOT_PRG] = (
        dataclasses.field(default_factory=FieldviewOpToFieldviewProg)
    )

    past_lint: workflow.Workflow[AOT_PRG, AOT_PRG] = dataclasses.field(
        default_factory=lambda: workflow.DataOnlyAdapter(past_linters.LinterFactory())
    )

    field_view_prog_args_transform: workflow.Workflow[AOT_PRG, AOT_PRG] = dataclasses.field(
        default=transform_prog_args
    )

    past_to_itir: workflow.Workflow[AOT_PRG, stages.AOTProgram] = dataclasses.field(
        default_factory=PastToItirAdapter
    )

    def step_order(
        self, inp: workflow.DataWithArgs[DSL_FOP | FOP | DSL_PRG | PRG | IT_PRG, ARGS | CARG]
    ) -> list[str]:
        steps: list[str] = []
        if isinstance(inp.args, ARGS):
            steps.append("aotify_args")
        match inp.data:
            case DSL_FOP():
                steps.extend(
                    [
                        "func_to_foast",
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                    ]
                )
            case FOP():
                steps.extend(
                    ["field_view_op_to_prog", "past_lint", "field_view_prog_args_transform"]
                )
            case DSL_PRG():
                steps.extend(["func_to_past", "past_lint", "field_view_prog_args_transform"])
            case PRG():
                steps.extend(["past_lint", "field_view_prog_args_transform"])
            case _:
                pass
        steps.append("past_to_itir")
        return steps


DEFAULT_TRANSFORMS: FieldopTransformWorkflow = FieldopTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class ExpBackend(backend.Backend):
    def __call__(
        self,
        program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = kwargs.pop("from_fieldop", None)
        # taking the offset provider out is not needed
        args, kwargs = convert_to_positional(program, *args, **kwargs)
        program_info = self.transforms_fop(
            workflow.DataWithArgs(
                data=program,  # type: ignore[arg-type] # TODO(ricoh): should go away when toolchain unified everywhere
                args=stages.CompileArgSpec.from_concrete_no_size(*args, **kwargs),
            )
        )
        # TODO(ricoh): get rid of executors altogether
        self.executor.otf_workflow(program_info)(*args, **kwargs)  # type: ignore[attr-defined]
