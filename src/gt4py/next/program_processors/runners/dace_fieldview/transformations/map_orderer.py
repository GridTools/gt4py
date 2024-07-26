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

from typing import Any, Optional, Sequence, Union

import dace
from dace import properties, transformation
from dace.sdfg import SDFG, SDFGState, nodes

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import utility as dace_fieldview_util


@properties.make_properties
class MapIterationOrder(transformation.SingleStateTransformation):
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
        The transformation does follow the rules outlines [here](https://hackmd.io/klvzLnzMR6GZBWtRU8HbDg#Requirements-on-SDFG)

    Todo:
        - Extend that different dimensions can be specified to be leading
            dimensions, with some priority mechanism.
        - Maybe also process the parameters to bring them in a canonical order.
    """

    leading_dim = properties.Property(
        dtype=gtx_common.Dimension,
        allow_none=True,
        desc="Dimension that should become the leading dimension.",
    )

    map_entry = transformation.transformation.PatternNode(nodes.MapEntry)

    def __init__(
        self,
        leading_dim: Optional[gtx_common.Dimension] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if leading_dim is not None:
            self.leading_dim = leading_dim

    @classmethod
    def expressions(cls) -> Any:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry)]

    def can_be_applied(
        self,
        graph: Union[SDFGState, SDFG],
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
        map_entry: nodes.MapEntry = self.map_entry
        map_params: Sequence[str] = map_entry.map.params
        map_var: str = dace_fieldview_util.get_map_variable(self.leading_dim)

        if map_var not in map_params:
            return False
        if map_params[-1] == map_var:  # Already at the correct location
            return False
        return True

    def apply(
        self,
        graph: Union[SDFGState, SDFG],
        sdfg: SDFG,
    ) -> None:
        """Performs the actual parameter reordering.

        The function will make the map variable, that corresponds to
        `self.leading_dim` the last map variable (this is given by the structure of
        DaCe's code generator).
        """
        map_entry: nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params
        map_var: str = dace_fieldview_util.get_map_variable(self.leading_dim)

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
