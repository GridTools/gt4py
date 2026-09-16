# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""An interface between DaCe's MapFusion and the one of GT4Py."""

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Callable, Optional, TypeAlias, Union

import dace
from dace import nodes as dace_nodes, properties as dace_properties
from dace.transformation import dataflow as dace_dftrans
from dace.transformation.dataflow import map_fusion_helper as dace_mfhelper


VerticalMapFusionCallback: TypeAlias = Callable[
    ["MapFusionVertical", dace_nodes.MapExit, dace_nodes.MapEntry, dace.SDFGState, dace.SDFG], bool
]
"""Callback used to influence the behaviour of `MapFusionVertical`.

The callback is called by `can_be_applied()` before any other check, thus it is not sure
if the two Maps can be fused at all. If `True` is returned the function will call the
`can_be_applied()` function of the actual Map fusion transformation. If `False` is
returned then `can_be_applied()` will immediately return `False`, i.e. the Maps will
not be checked any further.
It is very similar to the `HorizontalMapFusionCallback` but has different arguments.

It has the following arguments:
- The transformation object itself.
- The MaxExit node of the first Map.
- The MapEntry node of the second Map.
- The SDFGState in which the nodes where found.
- The SDFG itself.
"""

HorizontalMapFusionCallback: TypeAlias = Callable[
    ["MapFusionHorizontal", dace_nodes.MapEntry, dace_nodes.MapEntry, dace.SDFGState, dace.SDFG],
    bool,
]
"""Callback used to influence the behaviour of `MapFusionHorizontal`.

The callback is called by `can_be_applied()` before any other check, thus it is not sure
if the two Maps can be fused at all. If `True` is returned the function will call the
`can_be_applied()` function of the actual Map fusion transformation. If `False` is
returned then `can_be_applied()` will immediately return `False`, i.e. the Maps will
not be checked any further.
It is very similar to the `VerticalMapFusionCallback` but has different arguments.

It has the following arguments:
- The transformation object itself.
- The MaxEntry node of the first Map.
- The MapEntry node of the second Map.
- The SDFGState in which the nodes where found.
- The SDFG itself.
"""


@dace_properties.make_properties
class MapFusionVertical(dace_dftrans.MapFusionVertical):
    """GT4Py's Map fusion transformation for vertical Maps.

    Essentially the same as DaCe's `MapFusionVertical`, but with the following differences:
    - `strict_dataflow` is by default `False`, whereas in DaCe it is `True` for compatibility
        reasons.
    - It accepts the additional `check_fusion_callback`, which is called before the
        `can_be_applied()` function of the DaCe version is called. If it returns `False` then the
        fusion will be rejected.
    """

    _check_fusion_callback: Optional[VerticalMapFusionCallback]

    def __init__(
        self,
        strict_dataflow: bool = False,
        check_fusion_callback: Optional[VerticalMapFusionCallback] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            strict_dataflow=strict_dataflow,
            **kwargs,
        )
        self._check_fusion_callback = None
        if check_fusion_callback is not None:
            self._check_fusion_callback = check_fusion_callback

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        if self._check_fusion_callback is not None:
            if not self._check_fusion_callback(
                self, self.first_map_exit, self.second_map_entry, graph, sdfg
            ):
                return False
        return super().can_be_applied(graph, expr_index, sdfg, permissive)


@dace_properties.make_properties
class MapFusionHorizontal(dace_dftrans.MapFusionHorizontal):
    """GT4Py's Map fusion transformation for horizontal Maps.

    Essentially the same as DaCe's `MapFusionHorizontal`, but with the following difference:
    - It accepts the additional `check_fusion_callback`, which is called before the
        `can_be_applied()` function of the DaCe version is called. If it returns `False` then the
        fusion will be rejected.
    """

    _check_fusion_callback: Optional[HorizontalMapFusionCallback]

    def __init__(
        self,
        check_fusion_callback: Optional[HorizontalMapFusionCallback] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
        )
        self._check_fusion_callback = None
        if check_fusion_callback is not None:
            self._check_fusion_callback = check_fusion_callback

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        first_map_entry = self.first_parallel_map_entry
        second_map_entry = self.second_parallel_map_entry
        if self._check_fusion_callback is not None:
            if not self._check_fusion_callback(
                self, first_map_entry, second_map_entry, graph, sdfg
            ):
                return False

        # The pattern matches every pair of Maps in the state. The base class performs these
        #  checks too, but only after `is_parallel()`, which traverses the state twice. It also
        #  rejects a contradictory scope selection, which has to keep happening.
        if self.only_inner_maps and self.only_toplevel_maps:
            raise ValueError(
                "Only one of `only_inner_maps` and `only_toplevel_maps` is allowed per"
                f" `{type(self).__name__}` instance."
            )
        if first_map_entry.map.schedule != second_map_entry.map.schedule:
            return False
        scope_dict = graph.scope_dict()
        map_scope = scope_dict[first_map_entry]
        if scope_dict[second_map_entry] != map_scope:
            return False
        if self.only_toplevel_maps and map_scope is not None:
            return False
        if self.only_inner_maps and map_scope is None:
            return False
        if (
            dace_mfhelper.find_parameter_remapping(
                first_map=first_map_entry.map, second_map=second_map_entry.map
            )
            is None
        ):
            return False

        return super().can_be_applied(graph, expr_index, sdfg, permissive)
