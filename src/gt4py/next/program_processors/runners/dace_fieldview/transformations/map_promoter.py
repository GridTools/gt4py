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

from typing import Any, Mapping, Optional, Sequence, Union

import dace
from dace import properties, subsets, transformation
from dace.sdfg import SDFG, SDFGState, nodes


@properties.make_properties
class BaseMapPromoter(transformation.SingleStateTransformation):
    """Base transformation to add certain missing dimension to a map.

    By adding certain dimension to a map it will became possible to fuse them.
    This class acts as a base and the actual matching and checking must be
    implemented by a concrete implementation.

    In order to properly work, the parameters of `source_map` must be a strict
    superset of the ones of `map_to_promote`. Furthermore, this transformation
    builds upon the structure defined [here](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG).
    Thus it only checks the name of the parameters.

    To influence what to promote the user must implement the `map_to_promote()`
    and `source_map()` must be implemented. They have to return the map entry node.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        promote_vertical: If `True` promote vertical dimensions; `True` by default.
        promote_local: If `True` promote local dimensions; `True` by default.
        promote_horizontal: If `True` promote horizontal dimensions; `False` by default.

    Note:
        This ignores tiling.
        This only works with constant sized maps.
    """

    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are on the top level.",
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    promote_vertical = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` promote vertical dimensions.",
    )
    promote_local = properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` promote local dimensions.",
    )
    promote_horizontal = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` promote horizontal dimensions.",
    )

    def map_to_promote(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> nodes.MapEntry:
        """Returns the map entry that should be promoted."""
        raise NotImplementedError(f"{type(self).__name__} must implement 'map_to_promote'.")

    def source_map(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> nodes.MapEntry:
        """Returns the map entry that is used as source/template."""
        raise NotImplementedError(f"{type(self).__name__} must implement 'source_map'.")

    @classmethod
    def expressions(cls) -> Any:
        raise TypeError("You must implement 'expressions' yourself.")

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        promote_local: Optional[bool] = None,
        promote_vertical: Optional[bool] = None,
        promote_horizontal: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if only_inner_maps is not None:
            self.only_inner_maps = bool(only_inner_maps)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = bool(only_toplevel_maps)
        if promote_local is not None:
            self.promote_local = bool(promote_local)
        if promote_vertical is not None:
            self.promote_vertical = bool(promote_vertical)
        if promote_horizontal is not None:
            self.promote_horizontal = bool(promote_horizontal)
        if only_inner_maps and only_toplevel_maps:
            raise ValueError("You specified both `only_inner_maps` and `only_toplevel_maps`.")

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Perform some basic structural tests on the map.

        A subclass should call this function before checking anything else. If a
        subclass has not called this function, the behaviour will be undefined.
        The function checks:
        - If the map to promote is in the right scope (it is not required that
            the two maps are in the same scope).
        - If the parameter of the second map are compatible with each other.
        - If a dimension would be promoted that should not.
        """
        map_to_promote_entry: nodes.MapEntry = self.map_to_promote(state=graph, sdfg=sdfg)
        map_to_promote: nodes.Map = map_to_promote_entry.map
        source_map_entry: nodes.MapEntry = self.source_map(state=graph, sdfg=sdfg)
        source_map: nodes.Map = source_map_entry.map

        # Test the scope of the promotee.
        if self.only_inner_maps or self.only_toplevel_maps:
            scopeDict: Mapping[nodes.Node, Union[nodes.Node, None]] = graph.scope_dict()
            if self.only_inner_maps and (scopeDict[map_to_promote_entry] is None):
                return False
            if self.only_toplevel_maps and (scopeDict[map_to_promote_entry] is not None):
                return False

        # Test if the map ranges are compatible with each other.
        params_to_promote: list[str] | None = self.missing_map_params(
            map_to_promote=map_to_promote,
            source_map=source_map,
            be_strict=True,
        )
        if not params_to_promote:
            return False

        # Now we must check if there are dimensions that we do not want to promote.
        if (not self.promote_local) and any(
            param.endswith("__gtx_localdim") for param in params_to_promote
        ):
            return False
        if (not self.promote_vertical) and any(
            param.endswith("__gtx_vertical") for param in params_to_promote
        ):
            return False
        if (not self.promote_horizontal) and any(
            param.endswith("__gtx_horizontal") for param in params_to_promote
        ):
            return False

        return True

    def apply(self, graph: Union[SDFGState, SDFG], sdfg: SDFG) -> None:
        """Performs the Map Promoting.

        Add all parameters that `self.source_map` has but `self.map_to_promote`
        lacks to `self.map_to_promote` the range of these new dimensions is taken
        from the source map.
        The order of the parameters of these new dimensions is undetermined.
        """
        map_to_promote: nodes.Map = self.map_to_promote(state=graph, sdfg=sdfg).map
        source_map: nodes.Map = self.source_map(state=graph, sdfg=sdfg).map
        source_params: Sequence[str] = source_map.params
        source_ranges: subsets.Range = source_map.range

        missing_params: Sequence[str] = self.missing_map_params(  # type: ignore[assignment]  # Will never be `None`
            map_to_promote=map_to_promote,
            source_map=source_map,
            be_strict=False,
        )

        # Maps the map parameter of the source map to its index, i.e. which map
        #  parameter it is.
        map_source_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(source_params)}

        promoted_params = list(map_to_promote.params)
        promoted_ranges = list(map_to_promote.range.ranges)

        for missing_param in missing_params:
            promoted_params.append(missing_param)
            promoted_ranges.append(source_ranges[map_source_param_to_idx[missing_param]])

        # Now update the map properties
        #  This action will also remove the tiles
        map_to_promote.range = subsets.Range(promoted_ranges)
        map_to_promote.params = promoted_params

    def missing_map_params(
        self,
        map_to_promote: nodes.Map,
        source_map: nodes.Map,
        be_strict: bool = True,
    ) -> list[str] | None:
        """Returns the parameter that are missing in the map that should be promoted.

        The returned sequence is empty if they are already have the same parameters.
        The function will return `None` is promoting is not possible.

        Args:
            map_to_promote: The map that should be promoted.
            source_map: The map acting as template.
            be_strict: Ensure that the ranges that are already there are correct.
        """
        source_params_set: set[str] = set(source_map.params)
        curr_params_set: set[str] = set(map_to_promote.params)

        # The promotion can only work if the source map's parameters
        #  if a superset of the ones the map that should be promoted is.
        if not source_params_set.issuperset(curr_params_set):
            return None

        if be_strict:
            # Check if the parameters that are already in the map to promote have
            #  the same range as in the source map.
            source_ranges: subsets.Range = source_map.range
            curr_ranges: subsets.Range = map_to_promote.range
            curr_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(map_to_promote.params)}
            source_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(source_map.params)}
            for param_to_check in curr_params_set:
                curr_range = curr_ranges[curr_param_to_idx[param_to_check]]
                source_range = source_ranges[source_param_to_idx[param_to_check]]
                if curr_range != source_range:
                    return None
        return list(source_params_set - curr_params_set)


@properties.make_properties
class SerialMapPromoter(BaseMapPromoter):
    """This class promotes serial maps, such that they can be fused."""

    # Pattern Matching
    exit_first_map = transformation.transformation.PatternNode(nodes.MapExit)
    access_node = transformation.transformation.PatternNode(nodes.AccessNode)
    entry_second_map = transformation.transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        """Get the match expressions.

        The function generates two different match expression. The first match
        describes the case where the top map must be promoted, while the second
        case is the second/lower map must be promoted.
        """
        return [
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            ),
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            ),
        ]

    def map_to_promote(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> nodes.MapEntry:
        if self.expr_index == 0:
            # The first the top map will be promoted.
            return state.entry_node(self.exit_first_map)
        assert self.expr_index == 1

        # The second map will be promoted.
        return self.entry_second_map

    def source_map(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> nodes.MapEntry:
        """Returns the map entry that is used as source/template."""
        if self.expr_index == 0:
            # The first the top map will be promoted, so the second map is the source.
            return self.entry_second_map
        assert self.expr_index == 1

        # The second map will be promoted, so the first is used as source
        return state.entry_node(self.exit_first_map)
