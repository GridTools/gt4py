# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import Any, Dict, List, Set, Tuple

import dace
import dace.library
import dace.subsets

import gtc.common as common
import gtc.oir as oir
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gtc.common import LoopOrder
from gtc.dace.nodes import HorizontalExecutionLibraryNode, VerticalLoopLibraryNode
from gtc.dace.utils import (
    array_dimensions,
    get_access_collection,
    get_axis_bound_str,
    get_interval_range_str,
    get_tasklet_symbol,
)
from gtc.oir import Interval
from gtc.passes.oir_optimizations.utils import AccessCollector


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def visit_FieldAccess(self, node: oir.FieldAccess, *, is_target, targets):

        if (is_target or node.name in targets) and self.visit(node.offset) == "":
            targets.add(node.name)
            name = "__" + node.name
        else:
            name = node.name + "__" + self.visit(node.offset)
        if node.data_index:
            offset_str = str(node.data_index)
        else:
            offset_str = ""
        return name + offset_str

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
                common.NativeFunction.SIN: "dace.math.sin",
                common.NativeFunction.COS: "dace.math.cos",
                common.NativeFunction.TAN: "dace.math.tan",
                common.NativeFunction.ARCSIN: "asin",
                common.NativeFunction.ARCCOS: "acos",
                common.NativeFunction.ARCTAN: "atan",
                common.NativeFunction.SQRT: "dace.math.sqrt",
                common.NativeFunction.POW: "dace.math.pow",
                common.NativeFunction.EXP: "dace.math.exp",
                common.NativeFunction.LOG: "dace.math.log",
                common.NativeFunction.ISFINITE: "isfinite",
                common.NativeFunction.ISINF: "isinf",
                common.NativeFunction.ISNAN: "isnan",
                common.NativeFunction.FLOOR: "dace.math.ifloor",
                common.NativeFunction.CEIL: "ceil",
                common.NativeFunction.TRUNC: "trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    def visit_DataType(self, dtype: common.DataType, **kwargs: Any) -> str:
        if dtype == common.DataType.BOOL:
            return "dace.bool_"
        elif dtype == common.DataType.INT8:
            return "dace.int8"
        elif dtype == common.DataType.INT16:
            return "dace.int16"
        elif dtype == common.DataType.INT32:
            return "dace.int32"
        elif dtype == common.DataType.INT64:
            return "dace.int64"
        elif dtype == common.DataType.FLOAT32:
            return "dace.float32"
        elif dtype == common.DataType.FLOAT64:
            return "dace.float64"
        raise NotImplementedError("Not implemented DataType encountered.")

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

    LocalScalar = as_fmt("{name}: {dtype}")

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution):
        targets: Set[str] = set()
        return "\n".join([*self.visit(node.declarations), *self.visit(node.body, targets=targets)])

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        mask_str = ""
        indent = ""
        if node.mask is not None:
            mask_str = f"if {self.visit(node.mask, is_target=False, **kwargs)}:"
            indent = "    "
        body_code = self.visit(node.body, targets=kwargs["targets"])
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(node)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code


class OIRLibraryNodeExpander:
    def __init__(self, node, parent_state, parent_sdfg):
        self.res_sdfg = dace.SDFG(node.name + "_sdfg")
        self.res_state = self.res_sdfg.add_state(node.name + "_state", is_start_state=True)

        self.node = node
        self.parent_sdfg = parent_sdfg
        self.parent_state = parent_state

        self.context_subsets = {
            edge.dst_conn[len("IN_") :]: edge.data.subset
            for edge in parent_state.in_edges(node)
            if edge.dst_conn is not None
        }

        for edge in self.parent_state.out_edges(self.node):
            if edge.src_conn is None:
                continue
            name = edge.src_conn[len("OUT_") :]
            if name in self.context_subsets:
                self.context_subsets[name] = dace.subsets.union(
                    edge.data.subset, self.context_subsets[name]
                )
            else:
                self.context_subsets[name] = edge.data.subset
        self.origins = self.get_origins()

    def get_origins(self):
        raise NotImplementedError("Implement in subclass")

    def add_arrays(self):
        for name, subset in self.context_subsets.items():
            dtype = self.parent_sdfg.arrays[name].dtype
            strides = self.parent_sdfg.arrays[name].strides
            self.res_sdfg.add_array(
                name,
                shape=subset.bounding_box_size(),
                strides=strides,
                dtype=dtype,
            )

    def fix_context_memlets_and_get_nsdfg(self):

        in_connectors = set()
        out_connectors = set()
        for conn in self.node.in_connectors:
            in_connectors.add(conn[len("IN_") :])
        for conn in self.node.out_connectors:
            out_connectors.add(conn[len("OUT_") :])

        for edge in self.parent_state.in_edges(self.node):
            if edge.dst_conn is None:
                continue
            name = edge.dst_conn[len("IN_") :]
            edge.data.subset = self.context_subsets[name]
            edge.dst_conn = name
        for edge in self.parent_state.out_edges(self.node):
            if edge.src_conn is None:
                continue
            name = edge.src_conn[len("OUT_") :]
            edge.data.subset = self.context_subsets[name]
            edge.src_conn = name

        return dace.nodes.NestedSDFG(
            self.res_sdfg.name + "_nsdfg",
            self.res_sdfg,
            inputs=in_connectors,
            outputs=out_connectors,
        )

    def add_nodes_and_edges(self):
        raise NotImplementedError("Implement in Subclass")

    def expand(self):
        self.add_arrays()

        self.add_nodes_and_edges()
        res = self.fix_context_memlets_and_get_nsdfg()

        # inherit symbols from parent sdfg
        res.symbol_mapping = {s: s for s in self.res_sdfg.free_symbols}
        for s in list(self.res_sdfg.free_symbols):
            # res_sdfg already contains symbols for domain and strides where type is always int.
            # The type of API parameters still needs to be set.
            if s not in self.res_sdfg.symbols:
                self.res_sdfg.add_symbol(s, self.parent_sdfg.symbols[s])
        return res


class NaiveVerticalLoopExpander(OIRLibraryNodeExpander):

    node: VerticalLoopLibraryNode
    parent_state: dace.SDFGState
    parent_sdfg: dace.SDFG
    res_sdfg: dace.SDFG
    origins: Dict[str, Tuple[int, int, oir.AxisBound]]

    def get_ij_origins(
        self,
    ):

        origins: Dict[str, Tuple[int, int]] = {}

        for _, section in self.node.sections:
            for he in (
                ln
                for ln, _ in section.all_nodes_recursive()
                if isinstance(ln, HorizontalExecutionLibraryNode)
            ):
                access_collection = get_access_collection(he)

                for name, offsets in access_collection.offsets().items():
                    off: Tuple[int, int]
                    for off in offsets:
                        origin = (
                            -off[0] - he.iteration_space.i_interval.start.offset,
                            -off[1] - he.iteration_space.j_interval.start.offset,
                        )
                        if name not in origins:
                            origins[name] = origin
                        origins[name] = (
                            max(origins[name][0], origin[0]),
                            max(origins[name][1], origin[1]),
                        )
        return origins

    def get_k_origins(self):
        k_origs: Dict[str, oir.AxisBound] = {}
        for interval, section in self.node.sections:
            access_collection = get_access_collection(section)
            for name, offsets in access_collection.offsets().items():
                for off in offsets:
                    k_level = oir.AxisBound(
                        level=interval.start.level, offset=interval.start.offset + off[2]
                    )
                    k_orig = min(k_origs.get(name, k_level), k_level)
                    k_origs[name] = k_orig
        return k_origs

    def get_origins(self):
        ij_origins = self.get_ij_origins()
        k_origins = self.get_k_origins()
        return {name: (*ij_origins[name], k_origins[name]) for name in ij_origins.keys()}

    def get_mapped_subsets_dicts(self, interval: Interval, section: dace.SDFG):

        in_subsets = dict()
        out_subsets = dict()
        section_origins: Dict[str, Tuple[int, int]] = dict()
        min_k_offsets: Dict[str, int] = dict()
        for he in (
            ln
            for ln, _ in section.all_nodes_recursive()
            if isinstance(ln, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode))
        ):
            access_collection: AccessCollector.Result = get_access_collection(he)

            for name, offsets in access_collection.offsets().items():
                off: Tuple[int, int, int]
                for off in offsets:
                    origin = (
                        -off[0] - he.iteration_space.i_interval.start.offset,
                        -off[1] - he.iteration_space.j_interval.start.offset,
                    )
                    if name not in section_origins:
                        section_origins[name] = origin
                    if name not in min_k_offsets:
                        min_k_offsets[name] = off[2]
                    section_origins[name] = (
                        max(section_origins[name][0], origin[0]),
                        max(section_origins[name][1], origin[1]),
                    )
                    min_k_offsets[name] = min(min_k_offsets[name], off[2])
        access_collection = get_access_collection(section)
        for name, section_origin in section_origins.items():
            vl_origin = self.origins[name]
            shape = section.arrays[name].shape
            dimensions = array_dimensions(section.arrays[name])
            subset_strs = []
            idx = iter(range(3))
            if dimensions[0]:
                subset_strs.append(
                    "{i:+d}:{i:+d}+({I})".format(
                        i=vl_origin[0] - section_origin[0],
                        I=shape[next(idx)],
                    )
                )
            if dimensions[1]:
                subset_strs.append(
                    "{j:+d}:{j:+d}+({J})".format(
                        j=vl_origin[1] - section_origin[1], J=shape[next(idx)]
                    )
                )
            if dimensions[2]:
                subset_strs.append(
                    "k-({k_orig}){k:+d}:k-({k_orig}){k:+d}{K:+d}".format(
                        k_orig=get_axis_bound_str(vl_origin[2], "__K"),
                        k=min_k_offsets[name],
                        K=shape[next(idx)],
                    )
                )
            data_dims = shape[sum(dimensions) :]
            subset_strs.extend([f"0:{d}" for d in data_dims])
            subset_str = ",".join(subset_strs)
            if name in access_collection.read_fields():
                in_subsets[name] = subset_str
            if name in access_collection.write_fields():
                out_subsets[name] = subset_str
        return in_subsets, out_subsets


@dace.library.register_expansion(VerticalLoopLibraryNode, "naive")
class NaiveVerticalLoopExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def expansion(
        node: "VerticalLoopLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:
        from gtc.common import LoopOrder

        if node.loop_order == LoopOrder.PARALLEL:
            return ParallelNaiveVerticalLoopExpander(node, parent_state, parent_sdfg).expand()
        else:
            return SequentialNaiveVerticalLoopExpander(node, parent_state, parent_sdfg).expand()


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

    def add_nodes_and_edges(self):

        recent_state = self.res_state
        for interval, section in self.node.sections:
            loop_state = self.res_sdfg.add_state(section.name + "_state")
            _, _, recent_state = self.res_sdfg.add_loop(
                recent_state,
                loop_state,
                None,
                "k",
                *SequentialNaiveVerticalLoopExpander._get_loop_controls(
                    interval, self.node.loop_order
                ),
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
    def add_nodes_and_edges(self):
        # for each section
        # acc -> map over k -> nsdfg with HE's
        in_accesses = dict()
        out_accesses = dict()

        for interval, section in self.node.sections:

            interval_str = get_interval_range_str(interval, "__K")
            map_entry, map_exit = self.res_state.add_map(
                section.name + "_map", ndrange={"k": interval_str}
            )

            section_inputs = set()
            section_outputs = set()
            for acc in (
                n for n, _ in section.all_nodes_recursive() if isinstance(n, dace.nodes.AccessNode)
            ):
                if acc.access != dace.AccessType.WriteOnly:
                    if acc.data not in in_accesses:
                        in_accesses[acc.data] = self.res_state.add_read(acc.data)
                    section_inputs.add(acc.data)
                if acc.access != dace.AccessType.ReadOnly:
                    if acc.data not in out_accesses:
                        out_accesses[acc.data] = self.res_state.add_write(acc.data)
                    section_outputs.add(acc.data)

            nsdfg = self.res_state.add_nested_sdfg(
                sdfg=section,
                parent=None,
                inputs=section_inputs,
                outputs=section_outputs,
            )
            in_subsets, out_subsets = self.get_mapped_subsets_dicts(interval, section)
            if len(in_subsets) == 0:
                self.res_state.add_edge(map_entry, None, nsdfg, None, memlet=dace.memlet.Memlet())
            if len(out_subsets) == 0:
                self.res_state.add_edge(nsdfg, None, map_exit, None, memlet=dace.memlet.Memlet())
            for name, subset in in_subsets.items():
                self.res_state.add_memlet_path(
                    in_accesses[name],
                    map_entry,
                    nsdfg,
                    src_conn=None,
                    dst_conn=name,
                    memlet=dace.memlet.Memlet.simple(name, subset),
                )
            for name, subset in out_subsets.items():
                self.res_state.add_memlet_path(
                    nsdfg,
                    map_exit,
                    out_accesses[name],
                    src_conn=name,
                    dst_conn=None,
                    memlet=dace.memlet.Memlet.simple(name, subset, dynamic=False),
                )


class NaiveHorizontalExecutionExpander(OIRLibraryNodeExpander):
    def get_origins(self):
        access_collection: AccessCollector.Result = get_access_collection(self.node)

        origins = dict()
        for name, offsets in access_collection.offsets().items():
            origins[name] = access_collection.offsets()[name].pop()
            for off in offsets:
                origins[name] = (
                    min(origins[name][0], off[0]),
                    min(origins[name][1], off[1]),
                    min(origins[name][2], off[2]),
                )
            origins[name] = (
                -origins[name][0] - self.node.iteration_space.i_interval.start.offset,
                -origins[name][1] - self.node.iteration_space.j_interval.start.offset,
                -origins[name][2],
            )
        return origins

    def get_innermost_memlets(self):

        access_collection: AccessCollector.Result = get_access_collection(self.node)

        in_memlets = dict()
        for name, offsets in access_collection.read_offsets().items():
            dimensions = array_dimensions(self.parent_sdfg.arrays[name])
            data_dims = self.parent_sdfg.arrays[name].shape[sum(dimensions) :]

            for off in offsets:
                subset_strs = [
                    f"{var}{self.origins[name][dim] + off[dim]:+d}"
                    for dim, var in enumerate("ij0")
                    if dimensions[dim]
                ]
                subset_strs.extend(f"0:{dim}" for dim in data_dims)
                acc_name = get_tasklet_symbol(name, off, is_target=False)
                in_memlets[acc_name] = dace.memlet.Memlet.simple(name, ",".join(subset_strs))

        out_memlets = dict()
        for name in access_collection.write_fields():
            dimensions = array_dimensions(self.parent_sdfg.arrays[name])
            data_dims = self.parent_sdfg.arrays[name].shape[sum(dimensions) :]
            subset_strs = [
                f"{var}{self.origins[name][dim]:+d}"
                for dim, var in enumerate("ij0")
                if dimensions[dim]
            ]
            subset_strs.extend(f"0:{dim}" for dim in data_dims)
            acc_name = "__" + name
            out_memlets[acc_name] = dace.memlet.Memlet.simple(
                name,
                ",".join(subset_strs),
                dynamic=any(isinstance(stmt, oir.MaskStmt) for stmt in self.node.oir_node.body),
            )

        return in_memlets, out_memlets

    def add_nodes_and_edges(self):
        in_memlets, out_memlets = self.get_innermost_memlets()
        map_ranges = {
            "i": get_interval_range_str(self.node.iteration_space.i_interval, "__I"),
            "j": get_interval_range_str(self.node.iteration_space.j_interval, "__J"),
        }
        inputs = [name[len("IN_") :] for name in self.node.in_connectors]
        outputs = [name[len("OUT_") :] for name in self.node.out_connectors]
        input_nodes = {name: self.res_state.add_read(name) for name in inputs}
        output_nodes = {name: self.res_state.add_write(name) for name in outputs}

        self.res_state.add_mapped_tasklet(
            self.node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs=in_memlets,
            outputs=out_memlets,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            code=TaskletCodegen.apply(self.node.oir_node),
            external_edges=True,
        )


@dace.library.register_expansion(HorizontalExecutionLibraryNode, "naive")
class NaiveHorizontalExecutionExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def expansion(
        node: "HorizontalExecutionLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.SDFG:
        return NaiveHorizontalExecutionExpander(node, parent_state, parent_sdfg).expand()
