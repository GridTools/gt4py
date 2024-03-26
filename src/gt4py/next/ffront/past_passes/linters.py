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

import factory

from gt4py.next.ffront import gtcallable, stages as ffront_stages, transform_utils
from gt4py.next.otf import workflow


@workflow.make_step
def lint_misnamed_functions(
    inp: ffront_stages.PastProgramDefinition,
) -> ffront_stages.PastProgramDefinition:
    function_closure_vars = transform_utils._filter_closure_vars_by_type(
        inp.closure_vars, gtcallable.GTCallable
    )
    misnamed_functions = [
        f"{name} vs. {func.id}"
        for name, func in function_closure_vars.items()
        if name != func.__gt_itir__().id
    ]
    if misnamed_functions:
        raise RuntimeError(
            f"The following symbols resolve to a function with a mismatching name: {','.join(misnamed_functions)}."
        )
    return inp


@workflow.make_step
def lint_undefined_symbols(
    inp: ffront_stages.PastProgramDefinition,
) -> ffront_stages.PastProgramDefinition:
    undefined_symbols = [
        symbol.id for symbol in inp.past_node.closure_vars if symbol.id not in inp.closure_vars
    ]
    if undefined_symbols:
        raise RuntimeError(
            f"The following closure variables are undefined: {', '.join(undefined_symbols)}."
        )
    return inp


class LinterFactory(factory.Factory):
    class Meta:
        model = workflow.CachedStep

    step = lint_misnamed_functions.chain(lint_undefined_symbols)
    hash_function = ffront_stages.hash_past_program_definition
