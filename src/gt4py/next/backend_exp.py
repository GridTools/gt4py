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
import typing
from typing import Any, Generic, Optional

from gt4py.next import backend
from gt4py.next.ffront import (
    foast_to_itir,
    foast_to_past,
    func_to_foast,
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
                arg_types=tuple(arg.gt_type for arg in inp.args.args),
                kwarg_types={
                    k: v.gt_type for k, v in inp.args.kwargs.items() if k not in ["from_fieldop"]
                },
                closure_vars={inp.data.foast_node.id: ItirShim(inp.data, self.foast_to_itir)},
            )
        )
        return workflow.DataWithArgs(data=fieldview_program, args=inp.args)


@dataclasses.dataclass(frozen=True)
class PastToItirAdapter:
    step: workflow.Workflow[ffront_stages.AOTFieldviewProgramAst, stages.AOTProgram] = (
        dataclasses.field(default_factory=past_to_itir.PastToItir)
    )

    def __call__(
        self, inp: workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec]
    ) -> DataWithIteratorProgramMetadata[stages.AOTProgram]:
        aot_fvprog = ffront_stages.AOTFieldviewProgramAst(definition=inp.data, argspec=inp.args)
        aot_itprog = self.step(aot_fvprog)
        return DataWithIteratorProgramMetadata(
            data=aot_itprog,
            metadata=IteratorProgramMetadata(fieldview_program_type=inp.data.past_node.type),
        )


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
    past_to_itir: workflow[
        workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.CompileArgSpec],
        DataWithIteratorProgramMetadata[stages.AOTProgram],
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
            kwargs_subset = kwargs.copy()
            _ = kwargs_subset.pop("offset_provider")
            program_info = self.transforms_fop(
                workflow.DataWithArgs(
                    data=program,
                    args=stages.CompileArgSpec.from_concrete_no_size(*args, **kwargs_subset),
                )
            )
            compiled = self.executor.otf_workflow(program_info)
            compiled(*args, **kwargs)
