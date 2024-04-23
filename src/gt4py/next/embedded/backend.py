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

import warnings
from typing import Any

from gt4py._core import definitions as core_defs
from gt4py.next import (
    allocators as next_allocators,
    backend as next_backend,
    common,
    embedded as next_embedded,
    errors,
)
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.program_processors import modular_executor, processor_interface as ppi


class EmbeddedBackend(next_backend.Backend):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms_fop: next_backend.FieldopTransformWorkflow = next_backend.DEFAULT_FIELDOP_TRANSFORMS
    transforms_prog: next_backend.ProgramTransformWorkflow = next_backend.DEFAULT_PROG_TRANSFORMS

    def __call__(
        self,
        program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
        *args: tuple[Any],
        # offset_provider: dict[str, common.Dimension],
        **kwargs: dict[str, Any],
    ) -> None:
        match program:
            case ffront_stages.ProgramDefinition():
                offset_provider = kwargs.pop("offset_provider")
                return self.call_program(program, *args, offset_provider=offset_provider, **kwargs)
            case ffront_stages.FieldOperatorDefinition():
                if "from_fieldop" in kwargs:
                    kwargs.pop("from_fieldop")
                return self.call_fieldoperator(program, *args, **kwargs)

    def call_program(
        self,
        program: ffront_stages.ProgramDefinition,
        *args: tuple[Any],
        offset_provider: dict[str, common.Dimension],
        **kwargs: dict[str, Any],
    ) -> None:
        warnings.warn(
            UserWarning(
                f"Field View Program '{program.definition.__name__}': Using Python execution, consider selecting a perfomance backend."
            ),
            stacklevel=2,
        )
        with next_embedded.context.new_context(offset_provider=offset_provider) as ctx:
            # TODO(ricoh): check if rewriting still needed
            # rewritten_args, size_args, kwargs = past_process_args._process_args(
            #    self.past_stage.past_node, args, kwargs
            # )
            if "out" in kwargs:
                args = (*args, kwargs.pop("out"))
            return ctx.run(program.definition, *args, **kwargs)

    def call_fieldoperator(
        self,
        program: ffront_stages.FieldOperatorDefinition,
        *args: tuple[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        if not next_embedded.context.within_valid_context():
            if kwargs.get("offset_provider", None) is None:
                raise errors.MissingArgumentError(None, "offset_provider", True)
        attributes = program.attributes
        if attributes is not None and any(
            has_scan_op_attribute := [
                attribute in attributes for attribute in ["init", "axis", "forward"]
            ]
        ):
            assert all(has_scan_op_attribute)
            forward = attributes["forward"]
            init = attributes["init"]
            axis = attributes["axis"]
            op = embedded_operators.ScanOperator(program.definition, forward, init, axis)
        else:
            op = embedded_operators.EmbeddedOperator(program.definition)
        return embedded_operators.field_operator_call(op, args, kwargs)


default_embedded = EmbeddedBackend(
    executor=modular_executor.ModularExecutor(name="new_embedded", otf_workflow=None),
    allocator=None,
)
