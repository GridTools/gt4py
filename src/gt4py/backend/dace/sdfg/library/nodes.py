import copy

from typing import Mapping, List
import dace
import dace.subsets
from dace.properties import SymbolicProperty
import dace.library
import dace.data
from .expansions import StencilExpandTransformation
from gt4py import ir as gt_ir
from gt4py.definitions import Extent
from gt4py.backend.dace.sdfg.builder import MappedMemletInfo


@dace.properties.make_properties
class ApplyMethod:

    k_interval = dace.properties.Property(
        dtype=tuple,
        desc="The interval of the value k to which this sdfg"
        "is supposed to be mapped or looped.",
        default=(0, 0),
    )
    input_extents = dace.properties.DictProperty(
        key_type=str, value_type=Extent, default=Extent((0, 0), (0, 0), (0, 0))
    )
    sdfg = dace.properties.SDFGReferenceProperty(default=dace.SDFG("_"))

    def __init__(
        self,
        label,
        k_interval,
        read_accesses,
        write_accesses,
        arrays: Mapping[str, dace.data.Array],
        code,
        symbols,
    ):

        if code is None:
            code = ""
        if isinstance(code, str):
            code = dace.properties.CodeProperty.from_string(
                code, language=dace.dtypes.Language.Python
            )

        sdfg = dace.SDFG(label + "_tasklet_sdfg")
        state = sdfg.add_state(label + "_state")

        tasklet = state.add_tasklet(
            name=label + "_tasklet",
            inputs=set(read_accesses.keys()),
            outputs=set(write_accesses.keys()),
            code=code.as_string,
        )
        min_offsets = dict()
        max_offsets = dict()
        input_extents = dict()
        for name, acc in read_accesses.items():
            offset = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            if acc.outer_name not in min_offsets:
                input_extents[acc.outer_name] = Extent.from_offset(offset)
                min_offsets[acc.outer_name] = offset
                max_offsets[acc.outer_name] = offset
            min_offsets[acc.outer_name] = tuple(
                min(min_o, o) for min_o, o in zip(min_offsets[acc.outer_name], offset)
            )
            max_offsets[acc.outer_name] = tuple(
                max(max_o, o) for max_o, o in zip(max_offsets[acc.outer_name], offset)
            )
            input_extents[acc.outer_name] = input_extents[acc.outer_name].union(
                Extent.from_offset(offset)
            )
        for name, acc in write_accesses.items():
            offset = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            if acc.outer_name not in min_offsets:
                min_offsets[acc.outer_name] = offset
                max_offsets[acc.outer_name] = offset
            min_offsets[acc.outer_name] = tuple(
                min(min_o, o) for min_o, o in zip(min_offsets[acc.outer_name], offset)
            )
            max_offsets[acc.outer_name] = tuple(
                max(max_o, o) for max_o, o in zip(max_offsets[acc.outer_name], offset)
            )

        for name, array in arrays.items():
            array = copy.deepcopy(array)
            array.shape = tuple(
                1 + max_o - min_o
                for max_o, min_o in zip(
                    max_offsets.get(name, (0, 0, 0)), min_offsets.get(name, (0, 0, 0))
                )
            )
            sdfg.add_datadesc(name, array)

        read_accessors = dict()
        write_accessors = dict()

        for acc in read_accesses.values():
            if acc.outer_name not in read_accessors:
                read_accessors[acc.outer_name] = state.add_read(acc.outer_name)
        for acc in write_accesses.values():
            if acc.outer_name not in write_accessors:
                write_accessors[acc.outer_name] = state.add_write(acc.outer_name)

        for name, acc in read_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = ",".join(
                str(o - e)
                for o, e in zip(offset_tuple, min_offsets.get(acc.outer_name, (0, 0, 0)))
            )
            state.add_edge(
                read_accessors[acc.outer_name],
                None,
                tasklet,
                name,
                dace.memlet.Memlet.simple(
                    acc.outer_name, subset_str=subset_str, num_accesses=1, dynamic=acc.dynamic
                ),
            )
        for name, acc in write_accesses.items():
            offset_tuple = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            subset_str = ",".join(
                str(o - e)
                for o, e in zip(offset_tuple, min_offsets.get(acc.outer_name, (0, 0, 0)))
            )
            state.add_edge(
                tasklet,
                name,
                write_accessors[acc.outer_name],
                None,
                dace.memlet.Memlet.simple(
                    acc.outer_name, subset_str=subset_str, num_accesses=1, dynamic=acc.dynamic
                ),
            )
        for k, sym in symbols.items():
            sdfg.add_symbol(k, sym.dtype)
        self.sdfg = sdfg
        self.k_interval = tuple(dace.symbolic.pystr_to_symbolic(i) for i in k_interval)
        self.input_extents = input_extents


@dace.library.node
class StencilLibraryNode(dace.library.LibraryNode):
    implementations = {"loop": StencilExpandTransformation}
    default_implementation = "loop"

    iteration_order = dace.properties.Property(
        dtype=gt_ir.IterationOrder,
        allow_none=False,
        desc="gt4py.ir.IterationOrder",
        default=gt_ir.IterationOrder.PARALLEL,
    )
    read_accesses = dace.properties.Property(
        dtype=dict, default={}, desc="map local symbol to MappedMemletInfo"
    )
    write_accesses = dace.properties.Property(
        dtype=dict, default={}, desc="map local symbol to MappedMemletInfo"
    )
    inputs = dace.properties.Property(dtype=set, default=set(), desc="names of inputs")
    outputs = dace.properties.Property(dtype=set, default=set(), desc="names of outputs")
    ij_range = dace.properties.Property(
        dtype=tuple, allow_none=True, desc="range as subset descriptor"
    )
    intervals = dace.properties.ListProperty(
        element_type=ApplyMethod,
        allow_none=False,
        desc="List of `ApplyMethods`, holding a `k_interval` range and an sdfg of the form "
        "read_accessors -> tasklet -> write_accessors",
    )
    loop_order = dace.properties.Property(
        dtype=str, allow_none=False, default="IJK", desc="order of loops, permutation of 'IJK'"
    )

    def __init__(
        self,
        name,
        iteration_order=gt_ir.IterationOrder.PARALLEL,
        *,
        intervals=[],
        ij_range=None,
        implementation=None,
    ):

        self.inputs = set()
        self.outputs = set()
        for apply_method in intervals:
            input_accessors = [
                n
                for n, _ in apply_method.sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode)
                and n.access != dace.dtypes.AccessType.WriteOnly
            ]
            output_accessors = [
                n
                for n, _ in apply_method.sdfg.all_nodes_recursive()
                if isinstance(n, dace.nodes.AccessNode)
                and n.access != dace.dtypes.AccessType.ReadOnly
            ]
            self.inputs |= set(accessor.data for accessor in input_accessors)
            self.outputs |= set(accessor.data for accessor in output_accessors)

        super().__init__(
            name,
            inputs=set("IN_" + name for name in self.inputs),
            outputs=set("OUT_" + name for name in self.outputs),
        )

        self.iteration_order = iteration_order

        self.ij_range = ij_range
        self.intervals = intervals
        if implementation is not None:
            self.implementation = implementation
