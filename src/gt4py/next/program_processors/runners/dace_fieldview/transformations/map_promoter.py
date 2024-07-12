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

    Attributes:
        map_to_promote: This is the map entry that should be promoted, i.e. dimensions
            will be added such that its parameter matches `source_map`.
        source_map: The map entry node that describes how `map_to_promote` should
            look after the promotion.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.

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

    # Pattern Matching
    map_to_promote = transformation.transformation.PatternNode(nodes.MapEntry)
    source_map = transformation.transformation.PatternNode(nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        raise TypeError("You must implement 'expressions' yourself.")

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if only_inner_maps is not None:
            self.only_inner_maps = bool(only_inner_maps)
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = bool(only_toplevel_maps)
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
        """
        map_to_promote_entry: nodes.MapEntry = self.map_to_promote
        map_to_promote: nodes.Map = map_to_promote_entry.map
        source_map_entry: nodes.MapEntry = self.source_map
        source_map: nodes.Map = source_map_entry.map

        # Test the scope of the promotee.
        if self.only_inner_maps or self.only_toplevel_maps:
            scopeDict: Mapping[nodes.Node, Union[nodes.Node, None]] = graph.scope_dict()
            if self.only_inner_maps and (scopeDict[map_to_promote_entry] is None):
                return False
            if self.only_toplevel_maps and (scopeDict[map_to_promote_entry] is not None):
                return False

        # Test if the map ranges are compatible with each other.
        if not self.missing_map_params(
            map_to_promote=map_to_promote,
            source_map=source_map,
            be_strict=True,
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
        map_to_promote: nodes.Map = self.map_to_promote.map
        source_map: nodes.Map = self.source_map.map
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
    ) -> Sequence[str] | None:
        """Returns the parameter that are missing in the map that should be promoted.

        The returned sequence is empty if they are already have the same parameters.
        The function will return `None` is promoting is not possible.

        Args:
            map_to_promote: The map that should be promoted.
            source_map: The map acting as template.
            be_strict: Ensure that the ranges that are already there are correct.
        """
        source_params: set[str] = set(source_map.params)
        curr_params: set[str] = set(map_to_promote.params)

        # The promotion can only work if the source map's parameters
        #  if a superset of the ones the map that should be promoted is.
        if not source_params.issuperset(curr_params):
            return None

        if be_strict:
            # Check if the parameters that are already in the map to promote have
            #  the same range as in the source map.
            source_ranges: subsets.Range = source_map.range
            curr_ranges: subsets.Range = map_to_promote.range
            curr_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(map_to_promote.params)}
            source_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(source_map.params)}
            for param_to_check in curr_params:
                curr_range = curr_ranges[curr_param_to_idx[param_to_check]]
                source_range = source_ranges[source_param_to_idx[param_to_check]]
                if curr_range != source_range:
                    return None
        return list(source_params - curr_params)
