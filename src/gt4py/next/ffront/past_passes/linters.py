# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.ffront.stages import ConcretePASTProgramDef, PASTProgramDef
from gt4py.next.otf import toolchain, workflow


def lint_undefined_symbols(
    inp: ffront_stages.PASTProgramDef,
) -> ffront_stages.PASTProgramDef:
    undefined_symbols = [
        symbol.id for symbol in inp.past_node.closure_vars if symbol.id not in inp.closure_vars
    ]
    if undefined_symbols:
        raise RuntimeError(
            f"The following closure variables are undefined: {', '.join(undefined_symbols)}."
        )
    return inp


def linter_factory(
    cached: bool = True, adapter: bool = True
) -> workflow.Workflow[PASTProgramDef, PASTProgramDef]:
    wf: workflow.Workflow[PASTProgramDef, PASTProgramDef] = workflow.StepSequence(
        (lint_undefined_symbols,)
    )
    if cached:
        wf = workflow.CachedStep.in_memory(
            step=wf, input_fingerprinter=ffront_stages.semantic_fingerprinter
        )
    return wf


def adapted_linter_factory(
    **kwargs: Any,
) -> workflow.Workflow[ConcretePASTProgramDef, ConcretePASTProgramDef]:
    return toolchain.DataOnlyAdapter(linter_factory(**kwargs))
