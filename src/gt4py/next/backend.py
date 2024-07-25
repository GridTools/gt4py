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
from typing import Any, Generic

from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators
from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.ffront.stages import DSL_FOP, DSL_PRG, FOP, PRG
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, stages, workflow
from gt4py.next.program_processors import processor_interface as ppi


ARGS: typing.TypeAlias = arguments.JITArgs
CARG: typing.TypeAlias = arguments.CompileArgSpec
IT_PRG: typing.TypeAlias = itir.FencilDefinition


INPUT_DATA_T: typing.TypeAlias = DSL_FOP | FOP | DSL_PRG | PRG | IT_PRG
INPUT_PAIR_T: typing.TypeAlias = workflow.DataArgsPair[INPUT_DATA_T, ARGS | CARG]


@workflow.make_step
def foast_to_foast_closure(
    inp: workflow.DataArgsPair[ffront_stages.FoastOperatorDefinition, arguments.JITArgs],
) -> ffront_stages.FoastClosure:
    from_fieldop = inp.args.kwargs.pop("from_fieldop")
    debug = inp.args.kwargs.pop("debug", inp.data.debug)
    return ffront_stages.FoastClosure(
        foast_op_def=dataclasses.replace(inp.data, debug=debug),
        args=inp.args.args,
        kwargs=inp.args.kwargs,
        closure_vars={inp.data.foast_node.id: from_fieldop},
    )


@dataclasses.dataclass(frozen=True)
class Backend(Generic[core_defs.DeviceTypeT]):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms_fop: workflow.Workflow[INPUT_PAIR_T, stages.AOTProgram]
    transforms_prog: workflow.Workflow[INPUT_PAIR_T, stages.AOTProgram]

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
            aot_program = self.transforms_fop(
                workflow.DataArgsPair(program, args=arguments.JITArgs(args, kwargs))
            )
        else:
            aot_program = self.transforms_prog(
                workflow.DataArgsPair(program, arguments.JITArgs(args, kwargs))
            )
        self.executor(aot_program.data, *args, column_axis=aot_program.args.column_axis, **kwargs)

    @property
    def __name__(self) -> str:
        return getattr(self.executor, "__name__", None) or repr(self)

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
