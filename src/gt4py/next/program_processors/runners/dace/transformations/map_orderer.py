# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Union

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import gtir_sdfg_utils


def gt_set_iteration_order(
    sdfg: dace.SDFG,
    unit_strides_dim: Optional[
        Union[str, gtx_common.Dimension, list[Union[str, gtx_common.Dimension]]]
    ] = None,
    unit_strides_kind: Optional[gtx_common.DimensionKind] = None,
    validate: bool = True,
    validate_all: bool = False,
) -> Any:
    """Set the iteration order of the Maps correctly.

    The iteration order, while irrelevant from an SDFG point of view, is highly
    relevant in code and the fastest varying index ("inner most loop" in CPU or
    "x block dimension" in GPU) should be associated with the stride 1 dimension
    of the array.
    This function will reorder the map indexes such that this is the case.

    See the `MapIterationOrder` transformation for more.

    Args:
        sdfg: The SDFG to process.
        unit_strides_dims: List of dimensions that is considered to have unit strides.
        unit_strides_kind: All dimensions of this kind are considered to have unit
            strides.
        validate: Perform validation at the end of the function.
        validate_all: Perform validation also on intermediate steps.
    """
    return sdfg.apply_transformations_once_everywhere(
        MapIterationOrder(
            unit_strides_dims=unit_strides_dim,
            unit_strides_kind=unit_strides_kind,
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

    There are two ways to specify the unit stride dimensions. The first one is to
    specify `unit_strides_dim` at construction. It is possible to either pass a
    single dimension or a list of dimensions, in which case they are interpreted
    as priorities, the lower the index the higher its priority, see example below.
    Assuming we have the `unit_strides_dim=[EdgeDim, VertexDim]`, then we have the
    following:
    - `Map[EdgeDim, KDim, VertexDim]` -> `Map[KDim, VertexDim, EdgeDim]`.
    - `Map[VertexDim, KDim]` -> `Map[KDim, VertexDim]`.
    - `Map[EdgeDim, KDim]` -> `Map[KDim, EdgeDim]`.
    - `Map[CellDim, KDim]` -> `Map[CellDim, KDim]` (no modification).

    The second way is to specify `unit_strides_kind`, which is a GT4Py `DimensionKind`
    enum value. In that case the all dimensions that are associated to this kind
    will be ordered such that they are the inner most. If a Map has multiple
    dimensions of the same kind, then their relative order is undefined.

    In any case the order of the non matches dimensions is unspecific.

    Args:
        unit_strides_dims: GT4Py dimensions that are associated with the dimension
            that is supposed to have stride 1. If it is a list it is used as a ranking.
        unit_strides_kind: All dimensions of this kind have potential stride 1.

    Note:
        - The transformation does follow the rules outlined in
            [ADR0018](https://github.com/GridTools/gt4py/tree/main/docs/development/ADRs/0018-Canonical_SDFG_in_GT4Py_Transformations.md)
            especially rule 11, regarding the names. This is used to identify vertical
            and horizontal dimensions if `unit_strides_kind` is used.
        - String passed to `unit_strides_dim` have to match the name of the Map parameter
            exactly.

    Todo:
        - Maybe also process the parameters to bring them in a canonical order.
    """

    unit_strides_dims = dace_properties.ListProperty(
        element_type=str,
        allow_none=True,
        default=None,
        desc="Dimensions that should become the leading dimension.",
    )
    unit_strides_kind = dace_properties.Property(
        dtype=gtx_common.DimensionKind,
        allow_none=True,
        default=None,
        desc="Which kind of dimension should be made leading.",
    )
    map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    def __init__(
        self,
        unit_strides_dims: Optional[
            Union[str, gtx_common.Dimension, list[Union[str, gtx_common.Dimension]]]
        ] = None,
        unit_strides_kind: Optional[gtx_common.DimensionKind] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if unit_strides_dims is not None and unit_strides_kind is not None:
            raise ValueError("Specified both 'unit_strides_dims' and 'unit_strides_kind'.")
        elif unit_strides_dims is not None:
            if isinstance(unit_strides_dims, (gtx_common.Dimension, str)):
                unit_strides_dims = [unit_strides_dims]
            self.unit_strides_dims = [
                unit_strides_dim
                if isinstance(unit_strides_dim, str)
                else gtir_sdfg_utils.get_map_variable(unit_strides_dim)
                for unit_strides_dim in unit_strides_dims
            ]
        elif unit_strides_kind is not None:
            self.unit_strides_kind = unit_strides_kind
        else:
            raise ValueError("Neither specified 'unit_strides_kind' not 'unit_strides_dim'")

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
        `self.unit_strides_dim` the last map variable (this is given by the structure of
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

        new_map_params = (
            self._compute_map_param_order_unit_strides_dims()
            if self.unit_strides_dims is not None
            else self._compute_map_param_order_unit_strides_kind()
        )

        if new_map_params is None or map_params == new_map_params:
            return None

        initial_order: dict[str, int] = {map_param: i for i, map_param in enumerate(map_params)}
        return [initial_order[new_map_param] for new_map_param in new_map_params]

    def _compute_map_param_order_unit_strides_kind(self) -> Optional[list[str]]:
        """Compute the new Map order for the case a `unit_strides_kind` was specified."""
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params
        kind_to_look_for: str = "_" + str(self.unit_strides_kind)

        # Because of the way how GT4Py generates the names, we can just check if the
        #  parameter ends with kind keyword. We will now divide the parameters into
        #  two groups the one we care and the one that we ignore.
        map_params_to_order: list[str] = [
            map_param for map_param in map_params if map_param.endswith(kind_to_look_for)
        ]
        if not map_params_to_order:
            return None

        # The final parameter order is that the "leading ones" are right most.
        #  However, their order, if there are multiple of the same kind, is unspecific.
        new_map_params: list[str] = [
            map_param for map_param in map_params if map_param not in map_params_to_order
        ]
        new_map_params.extend(map_params_to_order)

        return new_map_params

    def _compute_map_param_order_unit_strides_dims(self) -> Optional[list[str]]:
        """Computes the new iteration order if concrete dimensions were specified."""
        map_entry: dace_nodes.MapEntry = self.map_entry
        map_params: list[str] = map_entry.map.params

        # Find all the parameters that we want to order.
        map_params_to_order: set[str] = {
            map_param for map_param in map_params if map_param in self.unit_strides_dims
        }
        if not map_params_to_order:
            return None

        # We start with all parameters that we ignore/do not care about.
        new_map_params: list[str] = [
            map_param for map_param in map_params if map_param not in self.unit_strides_dims
        ]

        # Because how code generation works the leading dimension must be the most
        #  right one. Because this is also `self.unit_strides_dims[0]` we have to process
        #  then in reverse order.
        for map_param_to_check in reversed(self.unit_strides_dims):
            if map_param_to_check in map_params_to_order:
                new_map_params.append(map_param_to_check)
        return new_map_params
