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

from typing import Any

from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.codegens.gtfn.gtfn_module import GTFNTranslationStep
from gt4py.next.program_processors.processor_interface import program_formatter
from gt4py.next.program_processors.runners import gtfn


@program_formatter
def format_cpp(program: itir.FencilDefinition, *args: Any, **kwargs: Any) -> str:
    # TODO(tehrengruber): This is a little ugly. Revisit.
    gtfn_translation = gtfn.GTFNBackendFactory().executor.otf_workflow.translation.inner
    assert isinstance(gtfn_translation, GTFNTranslationStep)
    return gtfn_translation.generate_stencil_source(
        program,
        offset_provider=kwargs.get("offset_provider", None),
        column_axis=kwargs.get("column_axis", None),
    )
