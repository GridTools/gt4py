# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Set, Tuple, Union

import dace.properties
import dace.subsets
from dace import SDFG, library

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc import common, oir
from gtc.common import LoopOrder
from gtc.oir import CacheDesc, HorizontalExecution, Interval, Stencil, VerticalLoop
from gtc.passes.oir_optimizations.utils import AccessCollector


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


@library.expansion
class NoLibraryNodeImplementation(dace.library.ExpandTransformation):
    environments: List = []


@library.expansion
class NaiveVerticalLoopExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def expansion(
        node: "VerticalLoopLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.SDFG:
        from gtc.common import LoopOrder

        if node.loop_order == LoopOrder.PARALLEL:
            return ParallelNaiveVerticalLoopExpander(node, parent_state, parent_sdfg).expand()
        else:
            return SequentialNaiveVerticalLoopExpander(node, parent_state, parent_sdfg).expand()


class NaiveVerticalLoopExpander:

    node: "VerticalLoopLibraryNode"
    parent_state: dace.SDFGState
    parent_sdfg: dace.SDFG
    res_sdfg: dace.SDFG
    origins: Dict[str, Tuple[int, int, oir.AxisBound]]

    def get_ij_origins(self, node: "VerticalLoopLibraryNode"):
        min_offsets: Dict[str, Tuple[int, int, int]] = {}
        for _, section in node.sections:
            access_collection = self._get_access_collection(section)
            min_offsets.update(
                {name: off.pop() for name, off in access_collection.offsets().items()}
            )
            for name, offsets in access_collection.offsets().items():
                off: Tuple[int, int, int]
                for off in offsets:
                    min_offsets[name] = (
                        min(min_offsets[name][0], off[0]),
                        min(min_offsets[name][1], off[1]),
                        min(min_offsets[name][2], off[2]),
                    )
        return {k: (-v[0], -v[1]) for k, v in min_offsets.items()}

    def get_k_origins(self, node: "VerticalLoopLibraryNode"):
        k_origs: Dict[str, oir.AxisBound] = {}
        for interval, section in node.sections:
            access_collection = self._get_access_collection(section)
            for name, offsets in access_collection.offsets().items():
                for off in offsets:
                    k_level = oir.AxisBound(
                        level=interval.start.level, offset=interval.start.offset + off[2]
                    )
                    k_orig = min(k_origs.get(name, k_level), k_level)
                    k_origs[name] = k_orig
        return k_origs

    def get_origins(self, node: "VerticalLoopLibraryNode"):
        ij_origins = self.get_ij_origins(node)
        k_origins = self.get_k_origins(node)
        return {name: (*ij_origins[name], k_origins[name]) for name in ij_origins.keys()}

    def __init__(
        self, node: "VerticalLoopLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ):
        self.node = node
        self.parent_state = parent_state
        self.parent_sdfg = parent_sdfg
        self.res_sdfg = dace.SDFG(node.name + "_sdfg")
        self._access_collection_cache: Dict[Any, AccessCollector.Result] = dict()
        self.origins = self.get_origins(node)

    def fix_context_memlets_and_get_nsdfg_add_arrays(self):
        in_connectors = set()
        out_connectors = set()
        for conn in self.node.in_connectors:
            in_connectors.add(conn[len("IN_") :])
        for conn in self.node.out_connectors:
            out_connectors.add(conn[len("OUT_") :])
        subsets = {
            edge.data.data: edge.data.subset
            for edge in self.parent_state.in_edges(self.node)
            if edge.data.data is not None
        }
        for edge in self.parent_state.out_edges(self.node):
            if edge.data.data in subsets:
                subsets[edge.data.data] = dace.subsets.union(
                    edge.data.subset, subsets[edge.data.data]
                )
            elif edge.data.data is not None:
                subsets[edge.data.data] = edge.data.subset
        for edge in self.parent_state.in_edges(self.node):
            if edge.dst_conn is not None:
                edge.data.subset = subsets[edge.data.data]
                edge.dst_conn = edge.dst_conn[len("IN_") :]
        for edge in self.parent_state.out_edges(self.node):
            if edge.src_conn is not None:
                edge.data.subset = subsets[edge.data.data]
                edge.src_conn = edge.src_conn[len("OUT_") :]

        for name, subset in subsets.items():
            dtype = self.parent_sdfg.arrays[name].dtype
            self.res_sdfg.add_array(
                name,
                shape=subset.bounding_box_size(),
                strides=tuple(
                    dace.symbolic.pystr_to_symbolic(f"__{name}_{var}_stride") for var in "IJK"
                ),
                dtype=dtype,
            )

        return dace.nodes.NestedSDFG(
            self.res_sdfg.name + "_nsdfg",
            self.res_sdfg,
            inputs=in_connectors,
            outputs=out_connectors,
        )

    def _get_access_collection(
        self, node: "Union[HorizontalExecutionLibraryNode, VerticalLoopLibraryNode, SDFG]"
    ) -> "AccessCollector.Result":
        if isinstance(node, dace.SDFG):
            res = AccessCollector.Result([])
            for node in node.states()[0].nodes():
                if isinstance(node, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                    collection = self._get_access_collection(node)
                    res._ordered_accesses.extend(collection._ordered_accesses)
            return res
        elif isinstance(node, HorizontalExecutionLibraryNode):
            if node.oir_node.id_ not in self._access_collection_cache:
                self._access_collection_cache[node.oir_node.id_] = AccessCollector.apply(
                    node.oir_node
                )
            return self._access_collection_cache[node.oir_node.id_]
        else:
            assert isinstance(node, VerticalLoopLibraryNode)
            res = AccessCollector.Result([])
            for _, sdfg in node.sections:
                collection = self._get_access_collection(sdfg)
                res._ordered_accesses.extend(collection._ordered_accesses)
            return res

    def get_mapped_subsets_dicts(self, interval: Interval, section: dace.SDFG):
        access_collection: AccessCollector.Result = self._get_access_collection(section)
        min_offsets = {name: off.pop() for name, off in access_collection.offsets().items()}

        in_subsets = dict()
        out_subsets = dict()

        for name, offsets in access_collection.offsets().items():
            for off in offsets:
                min_offsets[name] = (
                    min(min_offsets[name][0], off[0]),
                    min(min_offsets[name][1], off[1]),
                    min(min_offsets[name][2], off[2]),
                )
        for name, off in min_offsets.items():
            shape = section.arrays[name].shape

            subset_str = "{i:+d}:{i:+d}+({I}),{j:+d}:{j:+d}+({J}),k-({k_orig}){k:+d}:k-({k_orig}){k:+d}{K:+d}".format(
                i=self.origins[name][0] + off[0],
                j=self.origins[name][1] + off[1],
                k_orig=get_axis_bound_str(self.origins[name][2], "__K"),
                k=off[2],
                I=shape[0],
                J=shape[1],
                K=shape[2],
            )
            if name in access_collection.read_fields():
                in_subsets[name] = subset_str
            if name in access_collection.write_fields():
                out_subsets[name] = subset_str
        return in_subsets, out_subsets

    def _expand(self):
        raise NotImplementedError("Implement in Subclass")

    def expand(self):
        res = self.fix_context_memlets_and_get_nsdfg_add_arrays()
        self._expand()
        res.symbol_mapping = {s: s for s in self.res_sdfg.free_symbols}
        for s in list(self.res_sdfg.free_symbols):
            if s not in self.res_sdfg.symbols:
                self.res_sdfg.add_symbol(s, self.parent_sdfg.symbols[s])
        return res


class SequentialNaiveVerticalLoopExpander(NaiveVerticalLoopExpander):
    @staticmethod
    def _get_loop_controls(interval: Interval, loop_order: LoopOrder):
        if loop_order == LoopOrder.BACKWARD:
            initialize_expr = f"({get_axis_bound_str(interval.end, '__K')})-1"
            condition_expr = f"k>=({get_axis_bound_str(interval.start, '__K')})"
            increment_expr = "k-1"
        else:
            initialize_expr = f"{get_axis_bound_str(interval.start, '__K')}"
            condition_expr = f"k<({get_axis_bound_str(interval.end, '__K')})"
            increment_expr = "k+1"
        return initialize_expr, condition_expr, increment_expr

    def _expand(self):
        loop_order = self.node.loop_order
        recent_state = self.res_sdfg.add_state("start_state", is_start_state=True)

        # for each section
        for interval, section in self.node.sections:
            loop_state = self.res_sdfg.add_state(section.name + "_state")
            _, _, recent_state = self.res_sdfg.add_loop(
                recent_state,
                loop_state,
                None,
                "k",
                *SequentialNaiveVerticalLoopExpander._get_loop_controls(interval, loop_order),
            )

            in_accesses = dict()
            out_accesses = dict()
            in_subsets, out_subsets = self.get_mapped_subsets_dicts(interval, section)
            for acc in (
                n for n, _ in section.all_nodes_recursive() if isinstance(n, dace.nodes.AccessNode)
            ):
                if acc.access != dace.AccessType.WriteOnly:
                    if acc.data not in in_accesses:
                        in_accesses[acc.data] = loop_state.add_read(acc.data)
                if acc.access != dace.AccessType.ReadOnly:
                    if acc.data not in out_accesses:
                        out_accesses[acc.data] = loop_state.add_write(acc.data)
            nsdfg = loop_state.add_nested_sdfg(
                sdfg=section,
                parent=None,
                inputs={k for k in in_accesses},
                outputs={k for k in out_accesses},
            )
            for name, acc in in_accesses.items():
                loop_state.add_edge(
                    acc, None, nsdfg, name, dace.memlet.Memlet.simple(name, in_subsets[name])
                )
            for name, acc in out_accesses.items():
                loop_state.add_edge(
                    nsdfg, name, acc, None, dace.memlet.Memlet.simple(name, out_subsets[name])
                )


class ParallelNaiveVerticalLoopExpander(NaiveVerticalLoopExpander):
    def __init__(
        self, node: "VerticalLoopLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ):
        super().__init__(node, parent_state, parent_sdfg)

    def _expand(self):
        res_state = self.res_sdfg.add_state(self.node.name + "_state")
        # for each section
        # acc -> map over k -> nsdfg with HE's
        in_accesses = dict()
        out_accesses = dict()

        for interval, section in self.node.sections:

            interval_str = get_interval_range_str(interval, "__K")
            map_entry, map_exit = res_state.add_map(
                section.name + "_map", ndrange={"k": interval_str}
            )

            for acc in (
                n for n, _ in section.all_nodes_recursive() if isinstance(n, dace.nodes.AccessNode)
            ):
                if acc.access != dace.AccessType.WriteOnly:
                    if acc.data not in in_accesses:
                        in_accesses[acc.data] = res_state.add_read(acc.data)
                if acc.access != dace.AccessType.ReadOnly:
                    if acc.data not in out_accesses:
                        out_accesses[acc.data] = res_state.add_write(acc.data)

            nsdfg = res_state.add_nested_sdfg(
                sdfg=section,
                parent=None,
                inputs={k for k in in_accesses},
                outputs={k for k in out_accesses},
            )
            in_subsets, out_subsets = self.get_mapped_subsets_dicts(interval, section)
            if len(in_subsets) == 0:
                res_state.add_edge(map_entry, None, nsdfg, None, memlet=dace.memlet.Memlet())
            if len(out_subsets) == 0:
                res_state.add_edge(nsdfg, None, map_exit, None, memlet=dace.memlet.Memlet())
            for name, subset in in_subsets.items():
                res_state.add_memlet_path(
                    in_accesses[name],
                    map_entry,
                    nsdfg,
                    src_conn=None,
                    dst_conn=name,
                    memlet=dace.memlet.Memlet.simple(name, subset),
                )
            for name, subset in out_subsets.items():
                res_state.add_memlet_path(
                    nsdfg,
                    map_exit,
                    out_accesses[name],
                    src_conn=name,
                    dst_conn=None,
                    memlet=dace.memlet.Memlet.simple(name, subset, dynamic=False),
                )


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations = {"none": NoLibraryNodeImplementation, "naive": NaiveVerticalLoopExpansion}
    default_implementation = "naive"

    loop_order = dace.properties.Property(dtype=LoopOrder, default=None, allow_none=True)
    sections = dace.properties.ListProperty(
        element_type=Tuple[Interval, dace.SDFG], default=[], allow_none=False
    )
    caches = dace.properties.ListProperty(
        element_type=List[CacheDesc], default=[], allow_none=False
    )

    _dace_library_name = "oir.VerticalLoop"

    def __init__(
        self,
        name="unnamed_vloop",
        stencil: Stencil = None,
        oir_node: VerticalLoop = None,
        *args,
        **kwargs,
    ):

        from gtc.oir_to_dace import VerticalLoopSectionOirSDFGBuilder

        if oir_node is not None:
            name = oir_node.id_

        if stencil is not None:
            self.loop_order = oir_node.loop_order
            self.sections = [
                (
                    section.interval,
                    VerticalLoopSectionOirSDFGBuilder.build(section.id_, stencil, section),
                )
                for section in oir_node.sections
            ]
            self.caches = oir_node.caches

        super().__init__(name=name, *args, **kwargs)

    def validate(self, *args, **kwargs):
        for _, sdfg in self.sections:
            sdfg.validate()
        super().validate(*args, **kwargs)


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def visit_FieldAccess(self, node: oir.FieldAccess, *, is_target, targets):

        if (is_target or node.name in targets) and self.visit(node.offset) == "":
            targets.add(node.name)
            return "__" + node.name
        else:
            return node.name + "__" + self.visit(node.offset)

    def visit_CartesianOffset(self, node: common.CartesianOffset):
        res = []
        if node.i != 0:
            res.append(f'i{"m" if node.i<0 else "p"}{abs(node.i):d}')
        if node.j != 0:
            res.append(f'j{"m" if node.j<0 else "p"}{abs(node.j):d}')
        if node.k != 0:
            res.append(f'k{"m" if node.k<0 else "p"}{abs(node.k):d}')
        return "_".join(res)

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        right = self.visit(node.right, is_target=False, **kwargs)
        left = self.visit(node.left, is_target=True, **kwargs)
        return f"{left} = {right}"

    ScalarAccess = as_fmt("{name}")

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({true_expr} if {cond} else {false_expr})")

    def visit_BuiltInLiteral(self, builtin: common.BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == common.BuiltInLiteral.TRUE:
            return "True"
        elif builtin == common.BuiltInLiteral.FALSE:
            return "False"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_fmt("{dtype}({value})")
    Cast = as_fmt("{dtype}({expr})")

    def visit_NativeFunction(self, func: common.NativeFunction, **kwargs: Any) -> str:
        try:
            return {
                common.NativeFunction.ABS: "abs",
                common.NativeFunction.MIN: "min",
                common.NativeFunction.MAX: "max",
                common.NativeFunction.MOD: "fmod",
                common.NativeFunction.SIN: "math.sin",
                common.NativeFunction.COS: "math.cos",
                common.NativeFunction.TAN: "math.tan",
                common.NativeFunction.ARCSIN: "asin",
                common.NativeFunction.ARCCOS: "acos",
                common.NativeFunction.ARCTAN: "atan",
                common.NativeFunction.SQRT: "math.sqrt",
                common.NativeFunction.POW: "math.pow",
                common.NativeFunction.EXP: "math.exp",
                common.NativeFunction.LOG: "math.log",
                common.NativeFunction.ISFINITE: "isfinite",
                common.NativeFunction.ISINF: "isinf",
                common.NativeFunction.ISNAN: "isnan",
                common.NativeFunction.FLOOR: "floor",
                common.NativeFunction.CEIL: "ceil",
                common.NativeFunction.TRUNC: "trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    def visit_DataType(self, dtype: common.DataType, **kwargs: Any) -> str:
        if dtype == common.DataType.INT64:
            return "dace.int64"
        elif dtype == common.DataType.FLOAT64:
            return "dace.float64"
        elif dtype == common.DataType.FLOAT32:
            return "dace.float32"
        elif dtype == common.DataType.BOOL:
            return "dace.bool_"
        raise NotImplementedError("Not implemented NativeFunction encountered.")

    def visit_UnaryOperator(self, op: common.UnaryOperator, **kwargs: Any) -> str:
        if op == common.UnaryOperator.NOT:
            return " not "
        elif op == common.UnaryOperator.NEG:
            return "-"
        elif op == common.UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution):
        targets: Set[str] = set()
        mask_str = ""
        indent = ""
        if node.mask is not None:
            mask_str = f"if {self.visit(node.mask, is_target=False, targets=targets)}:"
            indent = "    "

        body_code = self.visit(node.body, targets=targets)
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(node)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code


@library.expansion
class NaiveHorizontalExecutionExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def code_generate(node):
        return TaskletCodegen.apply(node)

    @staticmethod
    def fix_context_memlets_and_get_nsdfg_add_arrays(node, res_sdfg, parent_sdfg, parent_state):

        in_connectors = set()
        out_connectors = set()
        for conn in node.in_connectors:
            in_connectors.add(conn[len("IN_") :])
        for conn in node.out_connectors:
            out_connectors.add(conn[len("OUT_") :])
        subsets = {
            edge.dst_conn[len("IN_") :]: edge.data.subset
            for edge in parent_state.in_edges(node)
            if edge.dst_conn is not None
        }
        for edge in parent_state.out_edges(node):
            if edge.src_conn is None:
                continue
            name = edge.src_conn[len("OUT_") :]
            if name in subsets:
                subsets[name] = dace.subsets.union(edge.data.subset, subsets[name])
            else:
                subsets[name] = edge.data.subset
        for edge in parent_state.in_edges(node):
            if edge.dst_conn is None:
                continue
            name = edge.dst_conn[len("IN_") :]
            edge.data.subset = subsets[name]
            edge.dst_conn = name
        for edge in parent_state.out_edges(node):
            if edge.src_conn is None:
                continue
            name = edge.src_conn[len("OUT_") :]
            edge.data.subset = subsets[name]
            edge.src_conn = name

        for name, subset in subsets.items():
            dtype = parent_sdfg.arrays[name].dtype
            res_sdfg.add_array(
                name,
                shape=subset.bounding_box_size(),
                strides=tuple(
                    dace.symbolic.pystr_to_symbolic(f"__{name}_{var}_stride") for var in "IJK"
                ),
                dtype=dtype,
            )

        return dace.nodes.NestedSDFG(
            res_sdfg.name + "_nsdfg",
            res_sdfg,
            inputs=in_connectors,
            outputs=out_connectors,
        )

    @staticmethod
    def _get_access_collection(
        node: "Union[HorizontalExecutionLibraryNode, VerticalLoopLibraryNode, SDFG]",
    ) -> "AccessCollector.Result":
        if isinstance(node, dace.SDFG):
            res = AccessCollector.Result([])
            for node in node.states()[0].nodes():
                if isinstance(node, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode)):
                    collection = NaiveHorizontalExecutionExpansion._get_access_collection(node)
                    res._ordered_accesses.extend(collection._ordered_accesses)
            return res
        elif isinstance(node, HorizontalExecutionLibraryNode):
            return AccessCollector.apply(node.oir_node)
        else:
            assert isinstance(node, VerticalLoopLibraryNode)
            res = AccessCollector.Result([])
            for _, sdfg in node.sections:
                collection = NaiveHorizontalExecutionExpansion._get_access_collection(sdfg)
                res._ordered_accesses.extend(collection._ordered_accesses)
            return res

    @staticmethod
    def get_innermost_memlets(node):

        access_collection: AccessCollector.Result = (
            NaiveHorizontalExecutionExpansion._get_access_collection(node)
        )
        min_offsets = {name: off.pop() for name, off in access_collection.offsets().items()}

        in_memlets = dict()
        out_memlets = dict()

        for name, offsets in access_collection.offsets().items():
            for off in offsets:
                min_offsets[name] = (
                    min(min_offsets[name][0], off[0]),
                    min(min_offsets[name][1], off[1]),
                    min(min_offsets[name][2], off[2]),
                )
        for name, offsets in access_collection.read_offsets().items():
            for off in offsets:
                subset_str = "i{i:+d},j{j:+d},{k:+d}".format(
                    i=-min_offsets[name][0] + off[0],
                    j=-min_offsets[name][1] + off[1],
                    k=-min_offsets[name][2] + off[2],
                )
                acc_name = name + "__"
                suffix = "_".join(
                    var + ("m" if o < 0 else "p") + f"{abs(o):d}"
                    for var, o in zip("ijk", off)
                    if o != 0
                )
                if suffix != "":
                    acc_name += suffix
                in_memlets[acc_name] = dace.memlet.Memlet.simple(name, subset_str)
        for name in access_collection.write_fields():
            subset_str = "i{i:+d},j{j:+d},{k:+d}".format(
                i=-min_offsets[name][0], j=-min_offsets[name][1], k=-min_offsets[name][2]
            )
            acc_name = "__" + name
            out_memlets[acc_name] = dace.memlet.Memlet.simple(
                name, subset_str, dynamic=node.oir_node.mask is not None
            )

        return in_memlets, out_memlets

    @staticmethod
    def expansion(
        node: "HorizontalExecutionLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.SDFG:

        res_sdfg = dace.SDFG(node.name + "_sdfg")
        res = NaiveHorizontalExecutionExpansion.fix_context_memlets_and_get_nsdfg_add_arrays(
            node, res_sdfg, parent_sdfg, parent_state
        )

        res_state = res_sdfg.add_state(node.name + "_state")
        in_memlets, out_memlets = NaiveHorizontalExecutionExpansion.get_innermost_memlets(node)
        map_ranges = {
            "i": get_interval_range_str(node.oir_node.iteration_space.i_interval, "__I"),
            "j": get_interval_range_str(node.oir_node.iteration_space.j_interval, "__J"),
        }
        inputs = [name[len("IN_") :] for name in node.in_connectors]
        outputs = [name[len("OUT_") :] for name in node.out_connectors]
        input_nodes = {name: res_state.add_read(name) for name in inputs}
        output_nodes = {name: res_state.add_write(name) for name in outputs}
        tasklet, _, _ = res_state.add_mapped_tasklet(
            node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs=in_memlets,
            outputs=out_memlets,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            code=NaiveHorizontalExecutionExpansion.code_generate(node.oir_node),
            external_edges=True,
        )
        res.symbol_mapping = {s: s for s in res_sdfg.free_symbols}
        for s in list(res_sdfg.free_symbols):
            if s not in res_sdfg.symbols:
                res_sdfg.add_symbol(s, parent_sdfg.symbols[s])
        return res


@library.node
class HorizontalExecutionLibraryNode(dace.nodes.LibraryNode):
    implementations = {
        "none": NoLibraryNodeImplementation,
        "naive": NaiveHorizontalExecutionExpansion,
    }
    default_implementation = "naive"

    oir_node = dace.properties.DataclassProperty(
        dtype=HorizontalExecution, default=None, allow_none=True
    )

    _dace_library_name = "oir.HorizontalExecution"

    def __init__(self, name="unnamed_vloop", oir_node: HorizontalExecution = None, *args, **kwargs):
        if oir_node is not None:
            name = oir_node.id_
            self.oir_node = oir_node

        super().__init__(name=name, *args, **kwargs)
