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

import dataclasses

import factory

from gt4py.next import common, config
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.ffront.gtcallable import GTCallable
from gt4py.next.ffront.past_to_itir import ProgramLowering
from gt4py.next.otf import stages, workflow
from gt4py.next.otf.stages import ProgramCall

from . import utils


@dataclasses.dataclass(frozen=True)
class PastToItir(workflow.ChainableWorkflowMixin):
    def __call__(self, inp: stages.PastClosure) -> ProgramCall:
        all_closure_vars = utils._get_closure_vars_recursively(
            inp.closure_vars)
        offsets_and_dimensions = utils._filter_closure_vars_by_type(
            all_closure_vars, FieldOffset, common.Dimension
        )
        grid_type = utils._deduce_grid_type(
            inp.grid_type, offsets_and_dimensions.values())

        gt_callables = utils._filter_closure_vars_by_type(
            all_closure_vars, GTCallable).values()
        lowered_funcs = [gt_callable.__gt_itir__()
                         for gt_callable in gt_callables]

        itir_program = ProgramLowering.apply(
            inp.past_node, function_definitions=lowered_funcs, grid_type=grid_type
        )

        if config.DEBUG or "debug" in inp.kwargs:
            devtools.debug(itir_program)

        return ProgramCall(
            itir_program,
            inp.args,
            inp.kwargs,
        )


class PastToItirFactory(factory.Factory):
    class Meta:
        model = PastToItir
