# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
from typing import Any, Iterable, Optional

from gt4py.next import common
from gt4py.next.ffront import fbuiltins
from gt4py.next.ffront.gtcallable import GTCallable


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
    offsets_and_dimensions: Iterable[fbuiltins.FieldOffset | common.Dimension],
) -> common.GridType:
    """
    Derive grid type from actually occurring dimensions and check against optional user request.

    Unstructured grid type is consistent with any kind of offset, cartesian
    is easier to optimize for but only allowed in the absence of unstructured
    dimensions and offsets.
    """

    def is_cartesian_offset(o: fbuiltins.FieldOffset) -> bool:
        return len(o.target) == 1 and o.source == o.target[0]

    deduced_grid_type = common.GridType.CARTESIAN
    for o in offsets_and_dimensions:
        if isinstance(o, fbuiltins.FieldOffset) and not is_cartesian_offset(o):
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
