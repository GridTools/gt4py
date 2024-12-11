# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from typing import Any, Mapping, Optional, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


@dace_properties.make_properties
class BaseMapPromoter(dace_transformation.SingleStateTransformation):
    """Base transformation to add certain missing dimension to a map.

    By adding certain dimension to a Map, it might became possible to use the Map
    in more transformations. This class acts as a base and the actual matching and
    checking must be implemented by a concrete implementation.
    But it provides some basic check functionality and the actual promotion logic.

    The transformation operates on two Maps, first the "source map". This map
    describes the Map that should be used as template. The second one is "map to
    promote". After the transformation the "map to promote" will have the same
    map parameter as the "source map" has.

    In order to properly work, the parameters of "source map" must be a strict
    superset of the ones of "map to promote". Furthermore, this transformation
    builds upon the structure defined [ADR0018](https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md)
    especially rule 11, regarding the names. Thus the function uses the names
    of the map parameters to determine what a local, horizontal, vertical or
    unknown dimension is. It also uses rule 12, therefore it will not perform
    a renaming and iteration variables must be a match.

    To influence what to promote the user must implement the `map_to_promote()`
    and `source_map()` function. They have to return the map entry node.

    The order of the parameter the map to promote has is unspecific, while the
    source map is not modified.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        promote_vertical: If `True` promote vertical dimensions, i.e. add them
            to the map to promote; `True` by default.
        promote_local: If `True` promote local dimensions, i.e. add them to the
            map to promote; `True` by default.
        promote_horizontal: If `True` promote horizontal dimensions, i.e. add
            them to the map to promote; `False` by default.
        promote_all: Do not impose any restriction on what to promote. The only
            reasonable value is `True` or `None`.

    Note:
        This ignores tiling.
    """

    only_toplevel_maps = dace_properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are on the top level.",
    )
    only_inner_maps = dace_properties.Property(
        dtype=bool,
        default=False,
        allow_none=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    promote_vertical = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` promote vertical dimensions.",
    )
    promote_local = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` promote local dimensions.",
    )
    promote_horizontal = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` promote horizontal dimensions.",
    )
    promote_all = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` perform any promotion. Takes precedence over all other selectors.",
    )

    def map_to_promote(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> dace_nodes.MapEntry:
        """Returns the map entry that should be promoted."""
        raise NotImplementedError(f"{type(self).__name__} must implement 'map_to_promote'.")

    def source_map(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> dace_nodes.MapEntry:
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
        promote_all: Optional[bool] = None,
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
        if promote_all is not None:
            self.promote_all = bool(promote_all)
            self.promote_horizontal = False
            self.promote_vertical = False
            self.promote_local = False
        if only_inner_maps and only_toplevel_maps:
            raise ValueError("You specified both `only_inner_maps` and `only_toplevel_maps`.")
        if not (
            self.promote_local
            or self.promote_vertical
            or self.promote_horizontal
            or self.promote_all
        ):
            raise ValueError(
                "You must select at least one class of dimension that should be promoted."
            )

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Perform some basic structural tests on the map.

        A subclass should call this function before checking anything else. If a
        subclass has not called this function, the behaviour will be undefined.
        The function checks:
        - If the map to promote is in the right scope.
        - If the parameter of the second map are compatible with each other.
        - If a dimension would be promoted that should not.
        """
        assert self.expr_index == expr_index
        map_to_promote_entry: dace_nodes.MapEntry = self.map_to_promote(state=graph, sdfg=sdfg)
        map_to_promote: dace_nodes.Map = map_to_promote_entry.map
        source_map_entry: dace_nodes.MapEntry = self.source_map(state=graph, sdfg=sdfg)
        source_map: dace_nodes.Map = source_map_entry.map

        # Test the scope of the promotee.
        #  Because of the nature of the transformation, it is not needed that the
        #  two maps are in the same scope. However, they should be in the same state
        #  to ensure that the symbols are the same and all. But this is guaranteed by
        #  the nature of this transformation (single state).
        if self.only_inner_maps or self.only_toplevel_maps:
            scope_dict: Mapping[dace_nodes.Node, Union[dace_nodes.Node, None]] = graph.scope_dict()
            if self.only_inner_maps and (scope_dict[map_to_promote_entry] is None):
                return False
            if self.only_toplevel_maps and (scope_dict[map_to_promote_entry] is not None):
                return False

        # Test if the map ranges and parameter are compatible with each other
        missing_map_parameters: list[str] | None = self.missing_map_params(
            map_to_promote=map_to_promote,
            source_map=source_map,
            be_strict=True,
        )
        if not missing_map_parameters:
            return False

        # We now know which dimensions we have to add to the promotee map.
        #  Now we must test if we are also allowed to make that promotion in the first place.
        if not self.promote_all:
            dimension_identifier: list[str] = []
            if self.promote_local:
                dimension_identifier.append("_gtx_localdim")
            if self.promote_vertical:
                dimension_identifier.append("_gtx_vertical")
            if self.promote_horizontal:
                dimension_identifier.append("_gtx_horizontal")
            if not dimension_identifier:
                return False
            for missing_map_param in missing_map_parameters:
                # Check if all missing parameter match a specified pattern. Note
                #  unknown iteration parameter, such as `__hansi_meier` will be
                #  rejected and can not be promoted.
                if not any(
                    missing_map_param.endswith(dim_identifier)
                    for dim_identifier in dimension_identifier
                ):
                    return False

        return True

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Performs the actual Map promoting.

        After this call the map to promote will have the same map parameters
        and ranges as the source map has. The function assumes that `can_be_applied()`
        returned `True`.
        """
        map_to_promote: dace_nodes.Map = self.map_to_promote(state=graph, sdfg=sdfg).map
        source_map: dace_nodes.Map = self.source_map(state=graph, sdfg=sdfg).map

        # The simplest implementation is just to copy the important parts.
        #  Note that we only copy the ranges and parameters all other stuff in the
        #  associated Map object is not modified.
        map_to_promote.params = copy.deepcopy(source_map.params)
        map_to_promote.range = copy.deepcopy(source_map.range)

    def missing_map_params(
        self,
        map_to_promote: dace_nodes.Map,
        source_map: dace_nodes.Map,
        be_strict: bool = True,
    ) -> list[str] | None:
        """Returns the parameter that are missing in the map that should be promoted.

        The returned sequence is empty if they are already have the same parameters.
        The function will return `None` is promoting is not possible.
        By setting `be_strict` to `False` the function will only check the names.

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
            source_ranges: dace_subsets.Range = source_map.range
            curr_ranges: dace_subsets.Range = map_to_promote.range
            curr_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(map_to_promote.params)}
            source_param_to_idx: dict[str, int] = {p: i for i, p in enumerate(source_map.params)}
            for param_to_check in curr_params_set:
                curr_range = curr_ranges[curr_param_to_idx[param_to_check]]
                source_range = source_ranges[source_param_to_idx[param_to_check]]
                # TODO(phimuell): Use simplify
                if curr_range != source_range:
                    return None
        return list(source_params_set - curr_params_set)


@dace_properties.make_properties
class SerialMapPromoter(BaseMapPromoter):
    """Promote a map such that it can be fused serially.

    A condition for fusing serial Maps is that they cover the same range. This
    transformation is able to promote a Map, i.e. adding the missing dimensions,
    such that the maps can be fused.
    For more information see the `BaseMapPromoter` class.

    Notes:
        The transformation does not perform the fusing on its one.

    Todo:
        The map should do the fusing on its own directly.
    """

    # Pattern Matching
    exit_first_map = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    entry_second_map = dace_transformation.PatternNode(dace_nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        """Get the match expressions.

        The function generates two match expressions. The first match describes
        the case where the top map must be promoted, while the second case is
        the second/lower map must be promoted.
        """
        return [
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            ),
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            ),
        ]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Tests if the Maps really can be fused."""

        # Test if the promotion could be done.
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        # Test if after the promotion the maps could be fused.
        if not self._test_if_promoted_maps_can_be_fused(graph, sdfg):
            return False

        return True

    def _test_if_promoted_maps_can_be_fused(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        """This function checks if the promoted maps can be fused by map fusion.

        This function assumes that `super().self.can_be_applied()` returned `True`.

        Args:
            state: The state in which we operate.
            sdfg: The SDFG we process.
        """
        first_map_exit: dace_nodes.MapExit = self.exit_first_map
        access_node: dace_nodes.AccessNode = self.access_node
        second_map_entry: dace_nodes.MapEntry = self.entry_second_map

        map_to_promote: dace_nodes.MapEntry = self.map_to_promote(state=state, sdfg=sdfg).map

        # Since we force a promotion of the map we have to store the old parameters
        #  of the map such that we can later restore them.
        org_map_to_promote_params = copy.deepcopy(map_to_promote.params)
        org_map_to_promote_ranges = copy.deepcopy(map_to_promote.range)

        try:
            # This will lead to a promotion of the map, this is needed that
            #  Map fusion can actually inspect them.
            self.apply(graph=state, sdfg=sdfg)
            if not gtx_transformations.MapFusionSerial.can_be_applied_to(
                sdfg=sdfg,
                expr_index=0,
                options={
                    "only_inner_maps": self.only_inner_maps,
                    "only_toplevel_maps": self.only_toplevel_maps,
                },
                map_exit_1=first_map_exit,
                intermediate_access_node=access_node,
                map_entry_2=second_map_entry,
            ):
                return False

        finally:
            # Restore the parameters of the map that we promoted before.
            map_to_promote.params = org_map_to_promote_params
            map_to_promote.range = org_map_to_promote_ranges

        return True

    def map_to_promote(
        self,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> dace_nodes.MapEntry:
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
    ) -> dace_nodes.MapEntry:
        """Returns the map entry that is used as source/template."""
        if self.expr_index == 0:
            # The first the top map will be promoted, so the second map is the source.
            return self.entry_second_map
        assert self.expr_index == 1

        # The second map will be promoted, so the first is used as source
        return state.entry_node(self.exit_first_map)
