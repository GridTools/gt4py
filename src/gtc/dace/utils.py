import re
from typing import TYPE_CHECKING, Any, Collection, Dict, Tuple, Union

import dace
import dace.data
from dace import SDFG, InterstateEdge

import eve
import gtc.oir as oir
from gtc.common import CartesianOffset, DataType, typestr_to_data_type
from gtc.oir import CartesianIterationSpace
from gtc.passes.oir_optimizations.utils import AccessCollector


if TYPE_CHECKING:
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
    from gtc.oir import VerticalLoopSection


def internal_symbols(sdfg: dace.SDFG):
    res = ["__I", "__J", "__K"]
    for name, array in sdfg.arrays.items():
        if isinstance(array, dace.data.Array):
            res.extend(
                [
                    f"__{name}_{var}_stride"
                    for idx, var in enumerate("IJK")
                    if array_dimensions(array)[idx]
                ]
            )
    return res


def validate_oir_sdfg(sdfg: dace.SDFG):

    from gtc.dace.nodes import VerticalLoopLibraryNode

    sdfg.validate()
    is_correct_node_types = all(
        isinstance(n, (dace.SDFGState, dace.nodes.AccessNode, VerticalLoopLibraryNode))
        for n, _ in sdfg.all_nodes_recursive()
    )
    is_correct_data_and_dtype = all(
        isinstance(array, dace.data.Array)
        and typestr_to_data_type(dace_dtype_to_typestr(array.dtype)) != DataType.INVALID
        for array in sdfg.arrays.values()
    )
    if not is_correct_node_types or not is_correct_data_and_dtype:
        raise ValueError("Not a valid OIR-level SDFG")


def dace_dtype_to_typestr(dtype: Any):
    if not isinstance(dtype, dace.typeclass):
        dtype = dace.typeclass(dtype)
    return dtype.as_numpy_dtype().str


def array_dimensions(array: dace.data.Array):
    stride_dims = [
        any(
            re.match(f"__.*_{k}_stride", str(sym))
            for st in array.strides
            for sym in st.free_symbols
        )
        for k in "IJK"
    ]
    shape_dims = [
        any(dace.symbol(f"__{k}") in sh.free_symbols for sh in array.shape) for k in "IJK"
    ]
    assert all(st == sh for st, sh in zip(stride_dims, shape_dims))
    return stride_dims


def get_axis_bound_str(axis_bound, var_name):
    from gtc.common import LevelMarker

    if axis_bound.level == LevelMarker.END:
        return f"{var_name}{axis_bound.offset:+d}"
    else:
        return f"{axis_bound.offset}"


def get_interval_range_str(interval, var_name):
    return "{}:{}".format(
        get_axis_bound_str(interval.start, var_name), get_axis_bound_str(interval.end, var_name)
    )


def get_axis_bound_diff_str(axis_bound1, axis_bound2, var_name: str):

    if axis_bound1 >= axis_bound2:
        tmp = axis_bound2
        axis_bound2 = axis_bound1
        axis_bound1 = tmp
        sign = "-"
    else:
        sign = ""

    if axis_bound1.level != axis_bound2.level:
        var = var_name
    else:
        var = ""
    return f"{sign}{var}{axis_bound2.offset-axis_bound1.offset:+d}"


def get_interval_length_str(interval, var_name):

    return "({})-({})".format(
        get_axis_bound_str(interval.end, var_name), get_axis_bound_str(interval.start, var_name)
    )


def get_vertical_loop_section_sdfg(section: "VerticalLoopSection") -> SDFG:
    sdfg = SDFG("VerticalLoopSection_" + str(id(section)))
    old_state = sdfg.add_state("start_state", is_start_state=True)
    for he in section.horizontal_executions:
        new_state = sdfg.add_state("HorizontalExecution_" + str(id(he)) + "_state")
        sdfg.add_edge(old_state, new_state, InterstateEdge())
        new_state.add_node(HorizontalExecutionLibraryNode(oir_node=he))

        old_state = new_state
    return sdfg


class OIRFieldRenamer(eve.NodeTranslator):
    def visit_FieldAccess(self, node: oir.FieldAccess):
        if node.name not in self._field_table:
            return node
        return oir.FieldAccess(
            name=self._field_table[node.name], offset=node.offset, dtype=node.dtype
        )

    def visit_ScalarAccess(self, node: oir.ScalarAccess):
        if node.name not in self._field_table:
            return node
        return oir.ScalarAccess(name=self._field_table[node.name], dtype=node.dtype)

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution):
        assert all(
            decl.name not in self._field_table or self._field_table[decl.name] == decl.name
            for decl in node.declarations
        )
        return node

    def __init__(self, field_table):
        self._field_table = field_table


def get_node_name_mapping(state: dace.SDFGState, node: dace.nodes.LibraryNode):

    name_mapping = dict()
    for edge in state.in_edges(node):
        if edge.dst_conn is not None:
            assert edge.dst_conn.startswith("IN_")
            internal_name = edge.dst_conn[len("IN_") :]
            outer_name = edge.data.data
            if internal_name not in name_mapping:
                name_mapping[internal_name] = outer_name
            else:
                msg = (
                    f"input and output of field '{internal_name}' to node'{node.name}' refer to "
                    + "different arrays"
                )
                assert name_mapping[internal_name] == outer_name, msg
    for edge in state.out_edges(node):
        if edge.src_conn is not None:
            assert edge.src_conn.startswith("OUT_")
            internal_name = edge.src_conn[len("OUT_") :]
            outer_name = edge.data.data
            if internal_name not in name_mapping:
                name_mapping[internal_name] = outer_name
            else:
                msg = (
                    f"input and output of field '{internal_name}' to node'{node.name}' refer to"
                    + "different arrays"
                )
                assert name_mapping[internal_name] == outer_name, msg
    return name_mapping


class CartesianIJIndexSpace(tuple):
    def __new__(cls, *args, **kwargs):
        return super(CartesianIJIndexSpace, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        msg = "CartesianIJIndexSpace must be a pair of pairs of integers."
        if not len(self) == 2:
            raise ValueError(msg)
        if not all(len(v) == 2 for v in self):
            raise ValueError(msg)
        if not all(isinstance(v[0], int) and isinstance(v[1], int) for v in self):
            raise ValueError(msg)

    @classmethod
    def from_offset(
        cls, offset: Union[Tuple[int, ...], CartesianOffset]
    ) -> "CartesianIJIndexSpace":
        if isinstance(offset, CartesianOffset):
            return cls((offset.i, offset.i), (offset.j, offset.j))
        return cls(((offset[0], offset[0]), (offset[1], offset[1])))

    @classmethod
    def from_iteration_space(
        cls, iteration_space: CartesianIterationSpace
    ) -> "CartesianIJIndexSpace":
        return cls(
            (
                (iteration_space.i_interval.start.offset, iteration_space.i_interval.end.offset),
                (iteration_space.j_interval.start.offset, iteration_space.j_interval.end.offset),
            )
        )

    def compose(
        self, other: Union["CartesianIterationSpace", "CartesianIJIndexSpace"]
    ) -> "CartesianIJIndexSpace":
        if isinstance(other, CartesianIterationSpace):
            other = CartesianIJIndexSpace.from_iteration_space(other)
        return CartesianIJIndexSpace(
            (
                (self[0][0] + other[0][0], (self[0][1] + other[0][1])),
                (self[1][0] + other[1][0], (self[1][1] + other[1][1])),
            )
        )

    def __or__(
        self, other: Union["CartesianIterationSpace", "CartesianIJIndexSpace"]
    ) -> "CartesianIJIndexSpace":
        if isinstance(other, CartesianIterationSpace):
            other = CartesianIJIndexSpace.from_iteration_space(other)
        return CartesianIJIndexSpace(
            (
                (min(self[0][0], other[0][0]), max(self[0][1], other[0][1])),
                (min(self[1][0], other[1][0]), max(self[1][1], other[1][1])),
            )
        )


def nodes_extent_calculation(
    nodes: Collection[Union["VerticalLoopLibraryNode", "HorizontalExecutionLibraryNode"]]
) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
    access_spaces: Dict[str, Tuple[Tuple[int, int], ...]] = dict()
    inner_nodes = []
    from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode

    for node in nodes:
        if isinstance(node, VerticalLoopLibraryNode):
            for _, section_sdfg in node.sections:
                for he in (
                    ln
                    for ln, _ in section_sdfg.all_nodes_recursive()
                    if isinstance(ln, dace.nodes.LibraryNode)
                ):
                    inner_nodes.append(he)
        else:
            assert isinstance(node, HorizontalExecutionLibraryNode)
            inner_nodes.append(node)
    for node in inner_nodes:
        access_collection = AccessCollector.apply(node.oir_node)
        iteration_space = node.oir_node.iteration_space
        if iteration_space is not None:
            for name, offsets in access_collection.offsets().items():
                for off in offsets:
                    access_extent = (
                        (
                            iteration_space.i_interval.start.offset + off[0],
                            iteration_space.i_interval.end.offset + off[0],
                        ),
                        (
                            iteration_space.j_interval.start.offset + off[1],
                            iteration_space.j_interval.end.offset + off[1],
                        ),
                    )
                    if name not in access_spaces:
                        access_spaces[name] = access_extent
                    access_spaces[name] = tuple(
                        (min(asp[0], ext[0]), max(asp[1], ext[1]))
                        for asp, ext in zip(access_spaces[name], access_extent)
                    )

    return {
        name: ((-asp[0][0], asp[0][1]), (-asp[1][0], asp[1][1]))
        for name, asp in access_spaces.items()
    }
