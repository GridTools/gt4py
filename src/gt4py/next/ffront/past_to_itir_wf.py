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
from typing import Any

import devtools
import factory

from gt4py.next import common, config
from gt4py.next.ffront import (
    fbuiltins,
    gtcallable,
    past_to_itir,
    transform_utils,
    type_specifications as ts_ffront,
)
from gt4py.next.otf import stages, workflow


@dataclasses.dataclass(frozen=True)
class PastToItir(workflow.ChainableWorkflowMixin):
    def __call__(self, inp: stages.PastClosure) -> stages.ProgramCall:
        all_closure_vars = transform_utils._get_closure_vars_recursively(inp.closure_vars)
        offsets_and_dimensions = transform_utils._filter_closure_vars_by_type(
            all_closure_vars, fbuiltins.FieldOffset, common.Dimension
        )
        grid_type = transform_utils._deduce_grid_type(
            inp.grid_type, offsets_and_dimensions.values()
        )

        gt_callables = transform_utils._filter_closure_vars_by_type(
            all_closure_vars, gtcallable.GTCallable
        ).values()
        lowered_funcs = [gt_callable.__gt_itir__() for gt_callable in gt_callables]

        itir_program = past_to_itir.ProgramLowering.apply(
            inp.past_node, function_definitions=lowered_funcs, grid_type=grid_type
        )

        if config.DEBUG or "debug" in inp.kwargs:
            devtools.debug(itir_program)

        return stages.ProgramCall(
            itir_program,
            inp.args,
            inp.kwargs | {"column_axis": _column_axis(all_closure_vars)},
        )


class PastToItirFactory(factory.Factory):
    class Meta:
        model = PastToItir


def _column_axis(all_closure_vars: dict[str, Any]) -> common.Dimension:
    # construct mapping from column axis to scan operators defined on
    #  that dimension. only one column axis is allowed, but we can use
    #  this mapping to provide good error messages.
    scanops_per_axis: dict[common.Dimension, str] = {}
    for name, gt_callable in transform_utils._filter_closure_vars_by_type(
        all_closure_vars, gtcallable.GTCallable
    ).items():
        if isinstance(
            (type_ := gt_callable.__gt_type__()),
            ts_ffront.ScanOperatorType,
        ):
            scanops_per_axis.setdefault(type_.axis, []).append(name)

    if len(scanops_per_axis.values()) == 0:
        return None

    if len(scanops_per_axis.values()) != 1:
        scanops_per_axis_strs = [
            f"- {dim.value}: {', '.join(scanops)}" for dim, scanops in scanops_per_axis.items()
        ]

        raise TypeError(
            "Only 'ScanOperator's defined on the same axis "
            + "can be used in a 'Program', found:\n"
            + "\n".join(scanops_per_axis_strs)
            + "."
        )

    return iter(scanops_per_axis.keys()).__next__()
