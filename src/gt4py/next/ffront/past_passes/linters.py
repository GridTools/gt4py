# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from gt4py.next.ffront import gtcallable, stages as ffront_stages, transform_utils
from gt4py.next.ffront.stages import AOT_PRG, PRG
from gt4py.next.otf import recipes, workflow


@workflow.make_step
def lint_misnamed_functions(
    inp: ffront_stages.PastProgramDefinition,
) -> ffront_stages.PastProgramDefinition:
    function_closure_vars = transform_utils._filter_closure_vars_by_type(
        inp.closure_vars, gtcallable.GTCallable
    )
    misnamed_functions = [
        f"{name} vs. {func.__gt_itir__().id}"
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


def linter_factory(cached: bool = True, adapter: bool = True) -> workflow.Workflow[PRG, PRG]:
    wf = lint_misnamed_functions.chain(lint_undefined_symbols)
    if cached:
        wf = workflow.CachedStep(step=wf, hash_function=ffront_stages.fingerprint_stage)
    return wf


def adapted_linter_factory(**kwargs: Any) -> workflow.Workflow[AOT_PRG, AOT_PRG]:
    return recipes.DataOnlyAdapter(linter_factory(**kwargs))
