# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors import program_formatter
from gt4py.next.program_processors.codegens.gtfn.gtfn_module import GTFNTranslationStep
from gt4py.next.program_processors.runners import gtfn


@program_formatter.program_formatter
def format_cpp(program: itir.Program, *args: Any, **kwargs: Any) -> str:
    # TODO(tehrengruber): This is a little ugly. Revisit.
    gtfn_translation = gtfn.GTFNBackendFactory().executor.translation
    assert isinstance(gtfn_translation, GTFNTranslationStep)
    return gtfn_translation.generate_stencil_source(
        program,
        offset_provider=kwargs.get("offset_provider", None),
        column_axis=kwargs.get("column_axis", None),
    )
