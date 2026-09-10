# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any, Optional, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes


@dace_properties.make_properties
class TrivialMapDimensionFolding(dace_transformation.SingleStateTransformation):
    """Replaces the parameter of a single iteration Map dimension by its value.

    A Map dimension such as `i = c:c+1:1` can only ever take the value `c`, but the
    Memlets inside its scope still refer to `i` symbolically. `MapFusionVertical`
    compares producer and consumer subsets with `Range.covers()`, which does not
    know the Map range, so a producer writing `a[c]` is not recognized as covering a
    consumer reading `a[i - o]` even though both denote the same element, and a legal
    fusion is rejected. Folding the value into the Memlets removes that blind spot.

    Note:
        Unlike DaCe's native `TrivialMapElimination` the dimension is not removed,
        only the single value is folded into the body of the Map.

    Args:
        only_toplevel_maps: Only process Maps that are on the top level.
    """

    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    only_toplevel_maps = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="See docs.",
    )

    def __init__(self, only_toplevel_maps: Optional[bool] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    @staticmethod
    def _single_iteration_parameters(map_: dace_nodes.Map) -> dict[str, Any]:
        """Returns the parameters of `map_` that can only take a single value."""
        return {
            param: rng[0]
            for param, rng in zip(map_.params, map_.range.ranges)
            if (rng[0] == rng[1]) == True  # noqa: E712 [true-false-comparison]  # SymPy comparison
        }

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        map_entry: dace_nodes.MapEntry = self.map_entry
        if self.only_toplevel_maps and graph.entry_node(map_entry) is not None:
            return False

        replacements = self._single_iteration_parameters(map_entry.map)
        if not replacements:
            return False

        # Only apply if a parameter is still referenced, otherwise the transformation
        #  would apply again and again on the same Map.
        scope = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True)
        for edge in scope.edges():
            if edge.data is not None and any(
                str(sym) in replacements for sym in edge.data.free_symbols
            ):
                return True
        for node in scope.nodes():
            if node is not map_entry and any(str(sym) in replacements for sym in node.free_symbols):
                return True
        return False

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        map_entry: dace_nodes.MapEntry = self.map_entry
        replacements = self._single_iteration_parameters(map_entry.map)
        scope = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True)

        # `replace_dict()` would also rewrite the Map's own parameters and range, which
        #  would drop the dimension, so they are restored afterwards.
        saved_params = copy.deepcopy(map_entry.map.params)
        saved_range = copy.deepcopy(map_entry.map.range)
        scope.replace_dict({param: str(value) for param, value in replacements.items()})
        map_entry.map.params = saved_params
        map_entry.map.range = saved_range
