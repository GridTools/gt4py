# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Sequence, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import utility as gtx_dace_fieldview_util


@dace_properties.make_properties
class MapIterationOrder(dace_transformation.SingleStateTransformation):
    """Modify the order of the iteration variables.

    The iteration order, while irrelevant from an SDFG point of view, is highly
    relevant in code, and the fastest varying index ("inner most loop" in CPU or
    "x block dimension" in GPU) should be associated with the stride 1 dimension
    of the array.
    This transformation will reorder the map indexes such that this is the case.

    While the place of the leading dimension is clearly defined, the order of the
    other loop indexes, after this transformation is unspecified.

    Args:
        leading_dim: A GT4Py dimension object that identifies the dimension that
            is supposed to have stride 1.

    Note:
        The transformation does follow the rules outlines in
        [ADR0018](https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md)
        especially rule 11, regarding the names.

    Todo:
        - Extend that different dimensions can be specified to be leading
            dimensions, with some priority mechanism.
        - Maybe also process the parameters to bring them in a canonical order.
    """

    leading_dim = dace_properties.Property(
        dtype=str,
        allow_none=True,
        desc="Dimension that should become the leading dimension.",
    )

    map_entry = dace_transformation.transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        leading_dim: Optional[Union[gtx_common.Dimension, str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(leading_dim, gtx_common.Dimension):
            self.leading_dim = gtx_dace_fieldview_util.get_map_variable(leading_dim)
        elif leading_dim is not None:
            self.leading_dim = leading_dim

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Test if the map can be reordered.

        Essentially the function checks if the selected dimension is inside the map,
        and if so, if it is on the right place.
        """

        if self.leading_dim is None:
            return False
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: Sequence[str] = map_entry.map.params
        map_var: str = self.leading_dim

        if map_var not in map_params:
            return False
        if map_params[-1] == map_var:  # Already at the correct location
            return False
        return True

    def apply(
        self,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
    ) -> None:
        """Performs the actual parameter reordering.

        The function will make the map variable, that corresponds to
        `self.leading_dim` the last map variable (this is given by the structure of
        DaCe's code generator).
        """
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params
        map_var: str = self.leading_dim

        # This implementation will just swap the variable that is currently the last
        #  with the one that should be the last.
        dst_idx = -1
        src_idx = map_params.index(map_var)

        for to_process in [
            map_entry.map.params,
            map_entry.map.range.ranges,
            map_entry.map.range.tile_sizes,
        ]:
            assert isinstance(to_process, list)
            src_val = to_process[src_idx]
            dst_val = to_process[dst_idx]
            to_process[dst_idx] = src_val
            to_process[src_idx] = dst_val
