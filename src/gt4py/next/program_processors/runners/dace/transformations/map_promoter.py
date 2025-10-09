# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import warnings
from typing import Any, Callable, Mapping, Optional, TypeAlias, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


MapPromotionCallBack: TypeAlias = Callable[
    [dace.SDFGState, dace.SDFG, dace_nodes.MapExit, dace_nodes.MapEntry, list[str]], bool
]
"""Callback for the `MapPromoter`.

After the `MapPromoter` has checked that the map would be promoted, the
callback is called. If the function returns `True` then transformation will
perform the promotion and in case of `False` the transformation will not
perform the promotion.
This allows user code to steer the promotion.

Args:
    state: The `SDFGState` on which the transformation operate.
    sdfg: The `SDFG` on which the transformation operate.
    first_map_exit: The `MapExit` node of the first Map, the one that is promoted.
    second_map_entry: The `MapEntry` node of the second Map.
    missing_map_parameters: The list of Map parameters that will be added to the
        first map.
"""


@dace_properties.make_properties
class MapPromoter(dace_transformation.SingleStateTransformation):
    """Promotes a Map such that it can be fused together.

    The transformation essentially matches `MapExit -> (Intermediate) -> MapEntry`
    and modifies the first Map such that it can be fused. The essentially checks
    if the parameter of the first Map are a subset of the second Map, in case
    they match they are checked for equality. If this is the case the
    transformation will add the missing parameters to the first map.
    It is possible to influence what kind of dimensions can be added.

    As an example, the transformation will turn the following code:
    ```python
    for i in dace.map[0:N]:
        a[i] = foo(i, ...)
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = bar(a[i], i, j, ...)
    ```
    into
    ```python
    for i, j in dace.map[0:N, 0:M]:
        a[i] = foo(i, ...)
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = bar(a[i], i, j, ...)
    ```
    which can be fused together by map fusion.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        promote_vertical: If `True` promote vertical dimensions, i.e. add them
            to the map to promote; `True` by default.
        promote_local: If `True` promote local dimensions, i.e. add them to the
            map to promote; `True` by default.
        promote_horizontal: If `True` promote horizontal dimensions, i.e. add
            them to the map to promote; `False` by default.
        promote_everything: Do not impose any restriction on what to promote. The only
            reasonable value is `True` or `False`.
        fuse_after_promotion: If `True`, the default, then fuse the two maps together
            immediately after promotion, i.e. inside `apply()`.
        promotion_callback: A callback function, see `MapPromotionCallBack`, that
            can be used to steer the promotion.

    Note:
        - The transformation will always promote the top Map never the lower Map.
        - This ignores tiling.

    Todo:
        Promotion in vertical direction, i.e. adding a vertical dimension is
        most likely all the time good, because it will create independent nodes
        that `LoopBlocking` can move out of the inner loop. However, promotion
        in horizontal, i.e. adding horizontal dimensions, is not necessarily
        good. Empirical observations have shown that it is most likely, but
        we should add a criteria to prevent the promotion in certain cases.
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
    promote_everything = dace_properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` perform any promotion. Takes precedence over all other selectors.",
    )
    fuse_after_promotion = dace_properties.Property(
        dtype=bool,
        default=True,
        desc="If `True` fuse the maps together immediately after promotion.",
    )

    _promotion_callback: Optional[MapPromotionCallBack]

    # Name of all data that is used at only one place. Is computed by the
    #  `FindSingleUseData` pass and be passed at construction time. Needed until
    #  [issue#1911](https://github.com/spcl/dace/issues/1911) has been solved.
    _single_use_data: dict[dace.SDFG, set[str]]

    # Pattern Matching
    exit_first_map = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    entry_second_map = dace_transformation.PatternNode(dace_nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            )
        ]

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        promote_local: Optional[bool] = None,
        promote_vertical: Optional[bool] = None,
        promote_horizontal: Optional[bool] = None,
        promote_everything: Optional[bool] = None,
        fuse_after_promotion: Optional[bool] = None,
        promotion_callback: Optional[MapPromotionCallBack] = None,
        *args: Any,
        single_use_data: dict[dace.SDFG, set[str]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if fuse_after_promotion is not None:
            self.fuse_after_promotion = fuse_after_promotion
        if only_inner_maps is not None:
            self.only_inner_maps = only_inner_maps
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = only_toplevel_maps
        if promote_local is not None:
            self.promote_local = promote_local
        if promote_vertical is not None:
            self.promote_vertical = promote_vertical
        if promote_horizontal is not None:
            self.promote_horizontal = promote_horizontal
        if promote_everything is not None:
            self.promote_everything = promote_everything
            self.promote_horizontal = False
            self.promote_vertical = False
            self.promote_local = False
        if only_inner_maps and only_toplevel_maps:
            raise ValueError("You specified both `only_inner_maps` and `only_toplevel_maps`.")
        if not (
            self.promote_local
            or self.promote_vertical
            or self.promote_horizontal
            or self.promote_everything
        ):
            raise ValueError(
                "You must select at least one class of dimension that should be promoted."
            )

        self._promotion_callback = None
        if promotion_callback is not None:
            self._promotion_callback = promotion_callback

        self._single_use_data = single_use_data

        # This flag is only for testing. It allows to bypass the check if the
        #  Maps can be fused after promotion.
        self._bypass_fusion_test = False

        if not self.fuse_after_promotion:
            warnings.warn(
                "Created a `MapPromoter` that does not fuse immediately, which might lead to borderline invalid SDFGs.",
                stacklevel=1,
            )

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        first_map_exit: dace_nodes.MapExit = self.exit_first_map
        second_map_entry: dace_nodes.MapEntry = self.entry_second_map

        if self.only_inner_maps or self.only_toplevel_maps:
            scope_dict: Mapping[dace_nodes.Node, Union[dace_nodes.Node, None]] = graph.scope_dict()
            if self.only_inner_maps and (scope_dict[second_map_entry] is None):
                return False
            if self.only_toplevel_maps and (scope_dict[second_map_entry] is not None):
                return False

        # Test if the map ranges and parameter are compatible with each other
        missing_map_parameters: list[str] | None = self._missing_map_params(
            map_to_promote=first_map_exit.map,
            source_map=second_map_entry.map,
            be_strict=True,
        )
        if not missing_map_parameters:
            return False

        # We now know which dimensions we have to add to the promotee map.
        #  Now we must test if we are also allowed to make that promotion in the first place.
        if not self.promote_everything:
            allowed_missing_dimension_suffixes: list[str] = []
            if self.promote_local:
                allowed_missing_dimension_suffixes.append("_gtx_localdim")
            if self.promote_vertical:
                allowed_missing_dimension_suffixes.append("_gtx_vertical")
            if self.promote_horizontal:
                allowed_missing_dimension_suffixes.append("_gtx_horizontal")
            if not allowed_missing_dimension_suffixes:
                return False
            for missing_map_param in missing_map_parameters:
                # Check if all missing parameter match a specified pattern. Note
                #  unknown iteration parameter, such as `__hansi_meier` will be
                #  rejected and can not be promoted.
                if not any(
                    missing_map_param.endswith(dim_identifier)
                    for dim_identifier in allowed_missing_dimension_suffixes
                ):
                    return False

        # It might be that the ranges that we add to the first Map from the second Map
        #  are empty. Which means that the second Map is never executed and after
        #  promotion, the first Map that was executed before would no longer be run.
        #  To prevent that we require that the number of iterations the second Map
        #  performs, at compile time, is larger than zero.
        second_map_iterations: Any = second_map_entry.map.range.num_elements()
        if str(second_map_iterations).isdigit():
            second_map_iterations = int(str(second_map_iterations))
            if second_map_iterations <= 0:
                return False
        elif hasattr(second_map_iterations, "free_symbols"):
            # According to [issue#2095](https://github.com/spcl/dace/issues/2095) DaCe is quite
            #  liberal concerning the positivity assumption, but in GT4Py this is not possible.
            second_map_iterations = second_map_iterations.subs(
                (
                    (sym, dace.symbol(sym.name, nonnegative=False))
                    for sym in list(second_map_iterations.free_symbols)
                )
            )
            if (second_map_iterations > 0) != True:  # noqa: E712 [true-false-comparison]  # SymPy fuzzy bools.
                return False
        else:
            warnings.warn(
                "Was unable to determine if the second Map ({second_map_entry}) is executed.",
                stacklevel=0,
            )
            return False

        if self._bypass_fusion_test:
            pass
        elif not self._test_if_promoted_maps_can_be_fused(graph, sdfg):
            return False

        if self._promotion_callback is not None:
            if not self._promotion_callback(
                graph, sdfg, first_map_exit, second_map_entry, missing_map_parameters
            ):
                return False

        return True

    def _promote_first_map(
        self,
        first_map_exit: dace_nodes.MapExit,
        second_map_entry: dace_nodes.MapEntry,
    ) -> None:
        first_map_exit.map.params = copy.deepcopy(second_map_entry.map.params)
        first_map_exit.map.range = copy.deepcopy(second_map_entry.map.range)

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        first_map_exit: dace_nodes.MapExit = self.exit_first_map
        access_node: dace_nodes.AccessNode = self.access_node
        second_map_entry: dace_nodes.MapEntry = self.entry_second_map

        # Now promote the second map such that it maps the first map.
        self._promote_first_map(first_map_exit, second_map_entry)

        if self.fuse_after_promotion:
            # Unlike in the `can_be_applied()` function we do specify here the right
            #  parameter, i.e. ensure that the intermediate can be removed. Because
            #  we can not pass the single use data, this will lead to a scan of the
            #  SDFG. However, we have to do it that way to get the desired result.
            gtx_transformations.MapFusionVertical.apply_to(
                sdfg=sdfg,
                options={
                    "only_inner_maps": self.only_inner_maps,
                    "only_toplevel_maps": self.only_toplevel_maps,
                    "require_exclusive_intermediates": True,
                    "require_all_intermediates": True,
                },
                # This will not run `MapFusionVertical.can_be_applied()`, thus we scan the
                #  SDFG only once instead of twice for every intermediate.
                verify=False,
                first_map_exit=first_map_exit,
                array=access_node,
                second_map_entry=second_map_entry,
            )

    def _missing_map_params(
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
                # TODO(phimuell): Use simplify?
                if curr_range != source_range:
                    return None
        return list(source_params_set - curr_params_set)

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

        # Since it is not possible to pass `self._single_use_data` to the Map fusion
        #  transformation, Map fusion would have to scan the SDFG on its own to
        #  figuring out if something is single use data or not. In order to avoid
        #  that we will now do the check ourselves. We not only check if the data is
        #  single use, but if it is only used (in this state) by the second Map. See
        #  bellow for why this is important.
        single_use_data = self._single_use_data[sdfg]
        for oedge in state.out_edges(first_map_exit):
            onode = oedge.dst
            if not isinstance(onode, dace_nodes.AccessNode):
                return False
            if not onode.desc(sdfg).transient:
                return False
            if onode.data not in single_use_data:
                return False
            if any(e.dst is not second_map_entry for e in state.out_edges(onode)):
                return False

        # Since we force a promotion of the map we have to store the old parameters
        #  of the map such that we can later restore them.
        first_map = first_map_exit.map
        org_first_map_params = copy.deepcopy(first_map.params)
        org_first_map_ranges = copy.deepcopy(first_map.range)

        try:
            # This will lead to a promotion of the map, this is needed that
            #  Map fusion can actually inspect them.
            self._promote_first_map(first_map_exit, second_map_entry)

            # Technically the promotion creates an invalid SDFG. Going back to the example
            #  in the doc string, after the promotion, but before the fusion, `a[i]` is
            #  written `M` times. This is not an issue per se, since each time the same
            #  value is written and after the fusion `a` will disappear, which ensures
            #  a valid SDFG. However, the removal of `a`, is only possible if the Map
            #  fusion classifies `a` as an "exclusive intermediate" (see `MapFusionVertical`).
            #  This is the reason why we have to set `require_all_intermediates`, to
            #  ensure that there are no "pure outputs" of the first Map.
            #  This also means that we would also have to specify `require_exclusive_intermediates`,
            #  because "shared intermediates" are not removed. But the test above has
            #  ensured that this is the case. As an optimization, to avoid that the
            #  Map fusion transformation scans the SDFG, we will specify `assume_always_shared`.
            #  This _is_ wrong, but since we only want to test if the Maps can be fused
            #  and this value has no influence on that outcome, we can specify it here.
            # TODO(phimuell): Once `single_use_data` can be passed, remove the optimization
            #   with `assume_always_shared`.
            if not gtx_transformations.MapFusionVertical.can_be_applied_to(
                sdfg=sdfg,
                options={
                    "only_inner_maps": self.only_inner_maps,
                    "only_toplevel_maps": self.only_toplevel_maps,
                    "require_all_intermediates": True,
                    "assume_always_shared": True,
                },
                first_map_exit=first_map_exit,
                array=access_node,
                second_map_entry=second_map_entry,
            ):
                return False

        finally:
            # Restore the parameters of the map that we promoted before.
            first_map.params = org_first_map_params
            first_map.range = org_first_map_ranges

        return True
