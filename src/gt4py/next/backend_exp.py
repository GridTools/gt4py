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
from typing import Any

from gt4py.next import backend
from gt4py.next.ffront import signature, stages as ffront_stages
from gt4py.next.otf import arguments, workflow


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
        args, kwargs = signature.convert_to_positional(program, *args, **kwargs)
        program_info = self.transforms_fop(
            workflow.DataArgsPair(
                data=program,
                args=arguments.CompileTimeArgs.from_concrete_no_size(*args, **kwargs),
            )
        )
        # TODO(ricoh): get rid of executors altogether
        self.executor.otf_workflow(program_info)(*args, **kwargs)  # type: ignore[attr-defined]
