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

import collections
import dataclasses
from collections.abc import Iterable

import factory

from gt4py.eve.extended_typing import Any, Optional
from gt4py.next import common
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.ffront.gtcallable import GTCallable
from gt4py.next.ffront.past_to_itir import ProgramLowering
from gt4py.next.ffront.source_utils import get_closure_vars_from_function
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, workflow
from gt4py.next.otf.stages import ProgramCall


def _get_closure_vars_recursively(closure_vars: dict[str, Any]) -> dict[str, Any]:
    all_closure_vars = collections.ChainMap(closure_vars)

    for closure_var in closure_vars.values():
        if isinstance(closure_var, GTCallable):
            # if the closure ref has closure refs by itself, also add them
            if child_closure_vars := closure_var.__gt_closure_vars__():
                all_child_closure_vars = _get_closure_vars_recursively(child_closure_vars)

                collisions: list[str] = []
                for potential_collision in set(closure_vars) & set(all_child_closure_vars):
                    if (
                        closure_vars[potential_collision]
                        != all_child_closure_vars[potential_collision]
                    ):
                        collisions.append(potential_collision)
                if collisions:
                    raise NotImplementedError(
                        f"Using closure vars with same name but different value "
                        f"across functions is not implemented yet. \n"
                        f"Collisions: '{',  '.join(collisions)}'."
                    )

                all_closure_vars = collections.ChainMap(all_closure_vars, all_child_closure_vars)
    return dict(all_closure_vars)


def _filter_closure_vars_by_type(closure_vars: dict[str, Any], *types: type) -> dict[str, Any]:
    return {name: value for name, value in closure_vars.items() if isinstance(value, types)}


def _deduce_grid_type(
    requested_grid_type: Optional[common.GridType],
    offsets_and_dimensions: Iterable[FieldOffset | common.Dimension],
) -> common.GridType:
    """
    Derive grid type from actually occurring dimensions and check against optional user request.

    Unstructured grid type is consistent with any kind of offset, cartesian
    is easier to optimize for but only allowed in the absence of unstructured
    dimensions and offsets.
    """

    def is_cartesian_offset(o: FieldOffset):
        return len(o.target) == 1 and o.source == o.target[0]

    deduced_grid_type = common.GridType.CARTESIAN
    for o in offsets_and_dimensions:
        if isinstance(o, FieldOffset) and not is_cartesian_offset(o):
            deduced_grid_type = common.GridType.UNSTRUCTURED
            break
        if isinstance(o, common.Dimension) and o.kind == common.DimensionKind.LOCAL:
            deduced_grid_type = common.GridType.UNSTRUCTURED
            break

    if (
        requested_grid_type == common.GridType.CARTESIAN
        and deduced_grid_type == common.GridType.UNSTRUCTURED
    ):
        raise ValueError(
            "'grid_type == GridType.CARTESIAN' was requested, but unstructured 'FieldOffset' or local 'Dimension' was found."
        )

    return deduced_grid_type if requested_grid_type is None else requested_grid_type


@dataclasses.dataclass(frozen=True)
class PastToItir(workflow.ChainableWorkflowMixin):
    def __call__(self, inp: stages.PastClosure) -> itir.FencilDefinition:
        closure_vars = _get_closure_vars_recursively(
            get_closure_vars_from_function(inp.definition)
        )
        offsets_and_dimensions = _filter_closure_vars_by_type(
            closure_vars, FieldOffset, common.Dimension
        )
        grid_type = _deduce_grid_type(inp.grid_type, offsets_and_dimensions.values())

        gt_callables = _filter_closure_vars_by_type(
            closure_vars, GTCallable
        ).values()
        lowered_funcs = [gt_callable.__gt_itir__() for gt_callable in gt_callables]
        return ProgramCall(
            ProgramLowering.apply(
                inp.past_node, function_definitions=lowered_funcs, grid_type=grid_type
            ),
            inp.args,
            inp.kwargs,
        )


class PastToItirFactory(factory.Factory):
    class Meta:
        model = PastToItir
