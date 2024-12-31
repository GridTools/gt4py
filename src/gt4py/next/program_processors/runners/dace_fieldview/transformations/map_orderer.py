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


def gt_set_iteration_order(
    sdfg: dace.SDFG,
    leading_dim: Optional[
        Union[str, gtx_common.Dimension, list[Union[str, gtx_common.Dimension]]]
    ] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> Any:
    """Set the iteration order of the Maps correctly.

    Modifies the order of the Map parameters such that `leading_dim`
    is the fastest varying one, the order of the other dimensions in
    a Map is unspecific. `leading_dim` should be the dimensions were
    the stride is one.

    Args:
        sdfg: The SDFG to process.
        leading_dim: The leading dimensions.
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.
    """
    return sdfg.apply_transformations_once_everywhere(
        MapIterationOrder(
            leading_dims=leading_dim,
        ),
        validate=validate,
        validate_all=validate_all,
    )


@dace_properties.make_properties
class MapIterationOrder(dace_transformation.SingleStateTransformation):
    """Modify the order of the iteration variables.

    The iteration order, while irrelevant from an SDFG point of view, is highly
    relevant in code and the fastest varying index ("inner most loop" in CPU or
    "x block dimension" in GPU) should be associated with the stride 1 dimension
    of the array.
    This transformation will reorder the map indexes such that this is the case.

    While the place of the leading dimension is clearly defined, the order of the
    other loop indexes, after this transformation is unspecified.

    The transformation accepts either a single dimension or a list of dimensions.
    In case a list is passed this is interpreted as priorities.
    Assuming we have the `leading_dim=[EdgeDim, VertexDim]`, then we have the
    following:
    - `Map[EdgeDim, KDim, VertexDim]` -> `Map[KDim, VertexDim, EdgeDim]`.
    - `Map[VertexDim, KDim]` -> `Map[KDim, VertexDim]`.
    - `Map[EdgeDim, KDim]` -> `Map[KDim, EdgeDim]`.
    - `Map[CellDim, KDim]` -> `Map[CellDim, KDim]` (no modification).

    Args:
        leading_dim: GT4Py dimensions that are associated with the dimension that is
            supposed to have stride 1. If it is a list it is used as a ranking.

    Note:
        The transformation does follow the rules outlines in
        [ADR0018](https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md)
        especially rule 11, regarding the names.

    Todo:
        - Extend that different dimensions can be specified to be leading
            dimensions, with some priority mechanism.
        - Maybe also process the parameters to bring them in a canonical order.
    """

    leading_dims = dace_properties.ListProperty(
        element_type=str,
        allow_none=True,
        default=None,
        desc="Dimensions that should become the leading dimension.",
    )

    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        leading_dims: Optional[
            Union[str, gtx_common.Dimension, list[Union[str, gtx_common.Dimension]]]
        ] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(leading_dims, (gtx_common.Dimension, str)):
            leading_dims = [leading_dims]
        if isinstance(leading_dims, list):
            self.leading_dims = [
                leading_dim
                if isinstance(leading_dim, str)
                else gtx_dace_fieldview_util.get_map_variable(leading_dim)
                for leading_dim in leading_dims
            ]

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
        if self.leading_dims is None:
            return False
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: Sequence[str] = map_entry.map.params
        processed_dims: set[str] = set(self.leading_dims)

        if not any(map_param in processed_dims for map_param in map_params):
            return False
        if self.compute_map_param_order() is None:
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
        map_object: dace_nodes.Map = self.map_entry.map
        new_map_params_order: list[int] = self.compute_map_param_order()  # type: ignore[assignment]  # Guaranteed to be not `None`.

        def reorder(what: list[Any]) -> list[Any]:
            assert isinstance(what, list)
            return [what[new_pos] for new_pos in new_map_params_order]

        map_object.params = reorder(map_object.params)
        map_object.range.ranges = reorder(map_object.range.ranges)
        map_object.range.tile_sizes = reorder(map_object.range.tile_sizes)

    def compute_map_param_order(self) -> Optional[list[int]]:
        """Computes the new iteration order of the matched map.

        The function returns a list, the value at index `i` indicates the old dimension
        that should be put at the new location. If the order is already correct then
        `None` is returned.
        """
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params
        org_mapping: dict[str, int] = {map_param: i for i, map_param in enumerate(map_params)}
        leading_dims: list[str] = self.leading_dims

        # We divide the map parameters into two groups, the one we care and the others.
        map_params_to_order: set[str] = {
            map_param for map_param in map_params if map_param in leading_dims
        }

        # If there is nothing to order, then we are done.
        if not map_params_to_order:
            return None

        # We start with all parameters that we ignore/do not care about.
        new_map_params: list[str] = [
            map_param for map_param in map_params if map_param not in leading_dims
        ]

        # Because how code generation works the leading dimension must be the most
        #  left one. Because this is also `self.leading_dims[0]` we have to process
        #  then in reverse order.
        for map_param_to_check in reversed(leading_dims):
            if map_param_to_check in map_params_to_order:
                new_map_params.append(map_param_to_check)
        assert len(map_params) == len(new_map_params)

        if map_params == new_map_params:
            return None

        return [org_mapping[new_map_param] for new_map_param in new_map_params]
