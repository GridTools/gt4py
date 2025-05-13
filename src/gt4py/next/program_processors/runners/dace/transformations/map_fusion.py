# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""An interface between DaCe's MapFusion and the one of GT4Py."""

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Callable, Optional, TypeAlias, TypeVar, Union

import dace
from dace import nodes as dace_nodes, properties as dace_properties

from gt4py.next.program_processors.runners.dace.transformations import (
    map_fusion_dace as dace_map_fusion,
)


_MapFusionType = TypeVar("_MapFusionType", bound="dace_map_fusion.MapFusion")

FusionTestCallback: TypeAlias = Callable[
    [_MapFusionType, dace_nodes.MapEntry, dace_nodes.MapEntry, dace.SDFGState, dace.SDFG, int], bool
]
"""Callback for the map fusion transformation to check if a fusion should be performed.

The callback returns `True` if the fusion should be performed and `False` if it
should be rejected. See also the description of GT4Py's MapFusion transformation for
more information.

The arguments are as follows:
- The transformation object that is active.
- The MapEntry node of the first map; exact meaning depends on if parallel or
    serial map fusion is performed.
- The MapEntry node of the second map; exact meaning depends on if parallel or
    serial map fusion is performed.
- The SDFGState that that contains the data flow.
- The SDFG that is processed.
- The expression index, see `expr_index` in `can_be_applied()` it is `0` for
    serial map fusion and `1` for parallel map fusion.
"""


@dace_properties.make_properties
class MapFusion(dace_map_fusion.MapFusion):
    """GT4Py's MapFusion transformation.

    It is a wrapper that adds some functionality to the transformation that is not
    present in the DaCe version of this transformation.
    There are three important differences when compared with DaCe's MapFusion:
    - In DaCe strict data flow is enabled by default, in GT4Py it is disabled by default.
    - In DaCe `MapFusion` only performs the fusion of serial maps by default. In GT4Py
        `MapFusion` will also perform parallel map fusion by default.
    - GT4Py accepts an additional argument `apply_fusion_callback`. This is a
        function that is called by the transformation, at the _beginning_ of
        `self.can_be_applied()`, i.e. before the transformation does any check if
        the maps can be fused. If this function returns `False`, `self.can_be_applied()`
        ends and returns `False`. In case the callback returns `True` the transformation
        will perform the usual steps to check if the transformation can apply or not.
        For the signature see `FusionTestCallback`.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        strict_dataflow: Strict dataflow mode should be used, it is disabled by default.
        assume_always_shared: Assume that all intermediates are shared.
        allow_serial_map_fusion: Allow serial map fusion, by default `True`.
        allow_parallel_fusion: Allow to merge parallel maps, by default `True`.
        only_if_common_ancestor: In parallel map fusion mode, only fuse if both maps
            have a common direct ancestor.
        apply_fusion_callback: The callback function that is used.

    Todo:
        Investigate ways of how to remove this intermediate layer. The main reason
        why we need it is the callback functionality, but it is not needed often
        and in these cases it might be solved differently.
    """

    _apply_fusion_callback: Optional[FusionTestCallback]

    def __init__(
        self,
        strict_dataflow: bool = False,
        allow_serial_map_fusion: bool = True,
        allow_parallel_map_fusion: bool = True,
        apply_fusion_callback: Optional[FusionTestCallback] = None,
        **kwargs: Any,
    ) -> None:
        self._apply_fusion_callback = None
        super().__init__(
            strict_dataflow=strict_dataflow,
            allow_serial_map_fusion=allow_serial_map_fusion,
            allow_parallel_map_fusion=allow_parallel_map_fusion,
            **kwargs,
        )
        if apply_fusion_callback is not None:
            self._apply_fusion_callback = apply_fusion_callback

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Performs basic checks if the maps can be fused.

        Args:
            map_entry_1: The entry of the first (in serial case the top) map.
            map_exit_2: The entry of the second (in serial case the bottom) map.
            graph: The SDFGState in which the maps are located.
            sdfg: The SDFG itself.
            permissive: Currently unused.
        """
        assert expr_index in [0, 1]

        # If the call back is given then proceed with it.
        if self._apply_fusion_callback is not None:
            if expr_index == 0:  # Serial MapFusion.
                first_map_entry: dace_nodes.MapEntry = graph.entry_node(self.first_map_exit)
                second_map_entry: dace_nodes.MapEntry = self.second_map_entry
            elif expr_index == 1:  # Parallel MapFusion
                first_map_entry = self.first_parallel_map_entry
                second_map_entry = self.second_parallel_map_entry
            else:
                raise NotImplementedError(f"Not implemented expression: {expr_index}")

            # Apply the call back.
            if not self._apply_fusion_callback(
                self,
                first_map_entry,
                second_map_entry,
                graph,
                sdfg,
                expr_index,
            ):
                return False

        # Now forward to the underlying implementation.
        return super().can_be_applied(
            graph=graph,
            expr_index=expr_index,
            sdfg=sdfg,
            permissive=permissive,
        )


@dace_properties.make_properties
class MapFusionSerial(MapFusion):
    """Wrapper around `MapFusion` that only supports serial map fusion.

    It overrides the `expressions()` class method. This means that the pattern
    matching system will not look for the parallel Maps that will be rejected
    anyway.

    Note:
        This class is needed to handle some stencils that have a lot of Maps, where
        serial Map fusion has to be run first to bring down the number of Maps.
        This splitting, having separate classes for parallel and serial Map fusion
        is where DaCe will go anyway.
    """

    @classmethod
    def expressions(cls) -> Any:
        return [
            dace.sdfg.utils.node_path_graph(cls.first_map_exit, cls.array, cls.second_map_entry)
        ]

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        assert "allow_serial_map_fusion" not in kwargs
        assert "allow_parallel_map_fusion" not in kwargs
        super().__init__(
            allow_serial_map_fusion=True,
            allow_parallel_map_fusion=False,
            **kwargs,
        )

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        assert expr_index == 0
        assert self.allow_parallel_map_fusion is False
        assert self.allow_serial_map_fusion

        for edge in graph.edges():
            edge.data.try_initialize(sdfg, graph, edge)

        # If the call back is given then proceed with it.
        if self._apply_fusion_callback is not None:
            first_map_entry: dace_nodes.MapEntry = graph.entry_node(self.first_map_exit)
            second_map_entry: dace_nodes.MapEntry = self.second_map_entry

            # Apply the call back.
            if not self._apply_fusion_callback(
                self,
                first_map_entry,
                second_map_entry,
                graph,
                sdfg,
                expr_index,
            ):
                return False

        return super().can_serial_map_fusion_be_applied(
            graph=graph,
            sdfg=sdfg,
        )

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        assert self.expr_index == 0
        assert self.allow_parallel_map_fusion is False
        assert self.allow_serial_map_fusion

        return super().apply_serial_map_fusion(
            graph=graph,
            sdfg=sdfg,
        )


@dace_properties.make_properties
class MapFusionParallel(MapFusion):
    """Wrapper around `MapFusion` that only supports parallel map fusion."""

    @classmethod
    def expressions(cls) -> Any:
        map_fusion_parallel_match = dace.sdfg.graph.OrderedMultiDiConnectorGraph()
        map_fusion_parallel_match.add_nodes_from(
            [cls.first_parallel_map_entry, cls.second_parallel_map_entry]
        )
        return [map_fusion_parallel_match]

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        assert "allow_serial_map_fusion" not in kwargs
        assert "allow_parallel_map_fusion" not in kwargs
        super().__init__(
            allow_serial_map_fusion=False,
            allow_parallel_map_fusion=True,
            **kwargs,
        )

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        assert expr_index == 0
        assert self.allow_parallel_map_fusion
        assert self.allow_serial_map_fusion is False

        for edge in graph.edges():
            edge.data.try_initialize(sdfg, graph, edge)

        # If the call back is given then proceed with it.
        if self._apply_fusion_callback is not None:
            first_map_entry = self.first_parallel_map_entry
            second_map_entry = self.second_parallel_map_entry

            # Apply the call back.
            if not self._apply_fusion_callback(
                self,
                first_map_entry,
                second_map_entry,
                graph,
                sdfg,
                expr_index,
            ):
                return False

        return super().can_parallel_map_fusion_be_applied(
            graph=graph,
            sdfg=sdfg,
        )

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        assert self.expr_index == 0
        assert self.allow_parallel_map_fusion
        assert self.allow_serial_map_fusion is False

        return super().apply_parallel_map_fusion(
            graph=graph,
            sdfg=sdfg,
        )
