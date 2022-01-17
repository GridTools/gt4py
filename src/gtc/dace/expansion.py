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

from typing import Any, Dict, List, Sequence, Set, Tuple

import dace
import dace.data
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
    CartesianIterationSpace,
    array_dimensions,
    get_access_collection,
    get_axis_bound_str,
    get_interval_range_str,
    get_tasklet_symbol,
)
from gtc.oir import Interval
from gtc.passes.oir_optimizations.utils import AccessCollector


def _get_offset_subset_str(origin, offset, nonflat_dimensions, symbols: Sequence[Any] = "ij0"):
    subset_strs = []
    for dim, var in enumerate(symbols):
        if nonflat_dimensions[dim]:
            subset_strs.append(f"({var})+({origin[dim]})+({offset[dim]})")

    return subset_strs


def _get_offset_suffix(node: Tuple[int, int, int]):
    res = []
    if node[0] != 0:
        res.append(f'i{"m" if node[0] < 0 else "p"}{abs(node[0]):d}')
    if node[1] != 0:
        res.append(f'j{"m" if node[1] < 0 else "p"}{abs(node[1]):d}')
    if node[2] is not None and node[2] != 0:
        res.append(f'k{"m" if node[2] < 0 else "p"}{abs(node[2]):d}')
    return "_".join(res)


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        origins,
        context_dimensions,
        nonflat_dimensions,
        is_target,
        targets,
        region_fields,
        variable_k_fields,
        index_symbols,
        **kwargs,
    ):
        idx_syms = list(index_symbols)
        if node.name in variable_k_fields:
            idx_syms[2] = "k"
        offset_for_suffix = list(node.offset.to_tuple())
        if node.name in region_fields:
            offset_for_suffix[0] = 0
            offset_for_suffix[1] = 0
        if node.name in variable_k_fields:
            offset_for_suffix[2] = 0
        elif is_target or node.name in targets:
            targets.add(node.name)
        name = get_tasklet_symbol(
            node.name, offset_for_suffix, (node.name in targets and self.visit(node.offset) == "")
        )

        if (
            node.name not in region_fields
            and node.name not in variable_k_fields
            and not node.data_index
        ):
            offset_str = ""
        else:
            offset_strs = []
            if node.name in region_fields or node.name in variable_k_fields:
                acc_name = get_tasklet_symbol(
                    node.name,
                    offset_for_suffix,
                    is_target,
                )
                offset = list(node.offset.to_tuple())
                if isinstance(node.offset, common.VariableKOffset):
                    varoffset = self.visit(
                        node.offset.k,
                        targets=targets,
                        is_target=False,
                        region_fields=region_fields,
                        variable_k_fields=variable_k_fields,
                        index_symbols=index_symbols,
                        context_dimensions=context_dimensions,
                        nonflat_dimensions=nonflat_dimensions,
                        origins=origins,
                        **kwargs,
                    )
                    offset[2] = f"{varoffset}"
                offset_strs += _get_offset_subset_str(
                    origins[node.name],
                    offset,
                    nonflat_dimensions=nonflat_dimensions[acc_name],
                    symbols=idx_syms,
                )

            if node.data_index:
                offset_strs += list(self.visit(node.data_index))
            offset_str = ",".join(offset_strs)
            if offset_str:
                offset_str = f"[{offset_str}]"
        return name + offset_str

    def visit_CartesianOffset(self, node: common.CartesianOffset):
        return _get_offset_suffix(node.to_tuple())

    def visit_VariableKOffset(self, node: common.VariableKOffset):
        return _get_offset_suffix(node.to_tuple())

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

    Literal = as_fmt("{value}")

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
                common.NativeFunction.SINH: "dace.math.sinh",
                common.NativeFunction.COSH: "dace.math.cosh",
                common.NativeFunction.TANH: "dace.math.tanh",
                common.NativeFunction.ARCSINH: "asinh",
                common.NativeFunction.ARCCOSH: "acosh",
                common.NativeFunction.ARCTANH: "atanh",
                common.NativeFunction.SQRT: "dace.math.sqrt",
                common.NativeFunction.POW: "dace.math.pow",
                common.NativeFunction.EXP: "dace.math.exp",
                common.NativeFunction.LOG: "dace.math.log",
                common.NativeFunction.GAMMA: "tgamma",
                common.NativeFunction.CBRT: "cbrt",
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

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, **kwargs):
        targets: Set[str] = set()
        return "\n".join(
            [
                *self.visit(node.declarations, **kwargs),
                *self.visit(node.body, targets=targets, **kwargs),
            ]
        )

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        mask_str = ""
        indent = ""
        if node.mask is not None:
            cond_str = self.visit(node.mask, is_target=False, **kwargs)
            if cond_str:
                mask_str = f"if {cond_str}:"
                indent = "    "
        body_code = self.visit(node.body, **kwargs)
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    def visit_HorizontalMask(
        self, node: oir.HorizontalMask, *, index_symbols, global_domain_symbols, **kwargs
    ):
        clauses: List[str] = []
        imin = get_axis_bound_str(node.i.start, global_domain_symbols[0])
        if imin:
            clauses.append(f"{index_symbols[0]} >= {imin}")
        imax = get_axis_bound_str(node.i.end, global_domain_symbols[0])
        if imax:
            clauses.append(f"{index_symbols[0]} < {imax}")
        jmin = get_axis_bound_str(node.j.start, global_domain_symbols[1])
        if jmin:
            clauses.append(f"{index_symbols[1]} >= {jmin}")
        jmax = get_axis_bound_str(node.j.end, global_domain_symbols[1])
        if jmax:
            clauses.append(f"{index_symbols[1]} < {jmax}")
        return " and ".join(clauses)

    @classmethod
    def apply(cls, node: oir.HorizontalExecution, **kwargs: Any) -> str:
        if not isinstance(node, oir.HorizontalExecution):
            raise ValueError("apply() requires oir.HorizontalExecution node")
        generated_code = super().apply(node, **kwargs)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code


class OIRLibraryNodeExpander:
    def __init__(self, node, parent_state, parent_sdfg, fix_context_memlets=True):
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
        self.origins = self.get_origins(compensate_regions=False)
        self.compensated_origins = self.get_origins(compensate_regions=True)
        self.fix_context_memlets = fix_context_memlets

    def get_origins(self, compensate_regions=False):
        raise NotImplementedError("Implement in subclass")

    def add_arrays(self):
        for name, subset in self.context_subsets.items():
            dtype = self.parent_sdfg.arrays[name].dtype
            strides = self.parent_sdfg.arrays[name].strides

            self.res_sdfg.add_array(
                name=name,
                shape=tuple(str(s) for s in subset.bounding_box_size()),
                strides=strides,
                dtype=dtype,
                storage=dace.StorageType.Default,
            )

    def fix_context_memlets_and_get_nsdfg(self):

        in_connectors = set()
        out_connectors = set()
        if self.fix_context_memlets:
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
        else:
            for conn in self.node.in_connectors:
                in_connectors.add(conn[len("IN_") :])
            for conn in self.node.out_connectors:
                out_connectors.add(conn[len("OUT_") :])

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
        for s in list(self.res_sdfg.free_symbols):
            # res_sdfg already contains symbols for domain and strides where type is always int.
            # The type of API parameters still needs to be set.
            if str(s) not in self.res_sdfg.symbols:
                self.res_sdfg.add_symbol(s, self.parent_sdfg.symbols.get(s, dace.int32))
        res.symbol_mapping = {s: s for s in self.res_sdfg.free_symbols}
        return res


class NaiveVerticalLoopExpander(OIRLibraryNodeExpander):

    node: VerticalLoopLibraryNode
    parent_state: dace.SDFGState
    parent_sdfg: dace.SDFG
    res_sdfg: dace.SDFG
    origins: Dict[str, Tuple[int, int, oir.AxisBound]]

    def add_arrays(self):
        super().add_arrays()
        for name, array in self.res_sdfg.arrays.items():
            if any(c.name == name and isinstance(c, oir.IJCache) for c in self.node.caches):
                array.storage = self.node.ijcache_storage_type

    def get_ij_origins(self, compensate_regions=False):

        origins: Dict[str, Tuple[int, int]] = {}

        for _, section in self.node.sections:
            for he in (
                ln
                for ln, _ in section.all_nodes_recursive()
                if isinstance(ln, HorizontalExecutionLibraryNode)
            ):
                access_collection = get_access_collection(he, compensate_regions=compensate_regions)

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
                        level=interval.start.level, offset=interval.start.offset + (off[2] or 0)
                    )
                    k_orig = min(k_origs.get(name, k_level), k_level)
                    k_origs[name] = k_orig
        return k_origs

    def get_origins(self, compensate_regions=False):
        ij_origins = self.get_ij_origins(compensate_regions=compensate_regions)
        k_origins = self.get_k_origins()
        return {name: (*ij_origins[name], k_origins[name]) for name in ij_origins.keys()}

    def get_mapped_subsets_dicts(self, interval: Interval, section: dace.SDFG):

        in_subsets = dict()
        out_subsets = dict()
        section_origins: Dict[str, Tuple[int, int]] = dict()
        min_k_offsets: Dict[str, int] = dict()
        max_k_offsets: Dict[str, int] = dict()
        for he in (
            ln
            for ln, _ in section.all_nodes_recursive()
            if isinstance(ln, (HorizontalExecutionLibraryNode, VerticalLoopLibraryNode))
        ):
            access_collection: AccessCollector.CartesianAccessCollection = get_access_collection(
                he, compensate_regions=True
            )

            for name, offsets in access_collection.offsets().items():
                off: Tuple[int, int, int]
                for off in offsets:
                    origin = (
                        -off[0] - he.iteration_space.i_interval.start.offset,
                        -off[1] - he.iteration_space.j_interval.start.offset,
                    )
                    if name not in section_origins:
                        section_origins[name] = origin
                    section_origins[name] = (
                        max(0, section_origins[name][0], origin[0]),
                        max(0, section_origins[name][1], origin[1]),
                    )

                    min_k_offsets.setdefault(name, off[2] or 0)
                    min_k_offsets[name] = min(min_k_offsets[name], off[2] or 0)

                    max_k_offsets.setdefault(name, off[2] or 0)
                    max_k_offsets[name] = max(max_k_offsets[name], off[2] or 0)

        access_collection = get_access_collection(section, compensate_regions=True)
        for name, section_origin in section_origins.items():
            vl_origin = self.compensated_origins[name]
            shape = section.arrays[name].shape
            dimensions = array_dimensions(section.arrays[name])
            subset_strs = []
            idx = iter(range(3))
            if dimensions[0]:
                subset_strs.append(
                    "{i:+d}:{i:+d}+({I})".format(
                        i=max(0, vl_origin[0]) - section_origin[0],
                        I=shape[next(idx)],
                    )
                )
            if dimensions[1]:
                subset_strs.append(
                    "{j:+d}:{j:+d}+({J})".format(
                        j=max(0, vl_origin[1]) - section_origin[1], J=shape[next(idx)]
                    )
                )
            if dimensions[2]:
                if any(
                    acc.offset[2] is None
                    for acc in access_collection.ordered_accesses()
                    if acc.field == name
                ):
                    subset_strs.append(
                        "({kstart})-({global_orig})+({min_off}):({kend})-({global_orig})+({max_off})".format(
                            kstart=get_axis_bound_str(interval.start, "__K"),
                            global_orig=get_axis_bound_str(vl_origin[2], "__K"),
                            min_off=min_k_offsets[name] or 0,
                            kend=get_axis_bound_str(interval.end, "__K"),
                            max_off=max_k_offsets[name] or 0,
                        )
                    )
                else:
                    subset_strs.append(
                        "k-({k_orig})+({k}):k-({k_orig})+({k})+({K})".format(
                            k_orig=get_axis_bound_str(vl_origin[2], "__K"),
                            k=min_k_offsets[name] or 0,
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
        self.res_sdfg.add_symbol("k", stype=dace.int32)
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
            nsdfg.symbol_mapping["k"] = f"k-({get_axis_bound_str(interval.start, '__K')})"
            nsdfg.symbol_mapping[
                "__K"
            ] = f"({get_axis_bound_str(interval.end, '__K')})-({get_axis_bound_str(interval.start, '__K')})"
            if "k" not in section.symbols:
                section.add_symbol("k", stype=dace.float32)
            if "__K" not in section.symbols:
                section.add_symbol("__K", stype=dace.float32)
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
                section.name + "_map",
                ndrange={"k": interval_str},
                schedule=self.node.map_schedule,
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
            nsdfg.symbol_mapping["k"] = f"k-({get_axis_bound_str(interval.start, '__K')})"
            nsdfg.symbol_mapping[
                "__K"
            ] = f"({get_axis_bound_str(interval.end, '__K')})-({get_axis_bound_str(interval.start, '__K')})"
            if "k" not in section.symbols:
                section.add_symbol("k", stype=dace.float32)
            if "__K" not in section.symbols:
                section.add_symbol("__K", stype=dace.float32)
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
    def get_origins(self, compensate_regions=False):
        access_collection: AccessCollector.CartesianAccessCollection = get_access_collection(
            self.node, compensate_regions=False
        )

        origins = dict()
        for acc in access_collection.ordered_accesses():
            if compensate_regions and acc.region is not None:
                offset = []
                iteration_space = self.node.iteration_space
                for dim, interval, iteration_interval in zip(
                    (0, 1),
                    (acc.region.i, acc.region.j),
                    (iteration_space.i_interval, iteration_space.j_interval),
                ):
                    if interval.start is None:
                        offset.append(min(acc.offset[dim], 0))
                    elif interval.start.level == common.LevelMarker.START:
                        if acc.offset[dim] > 0:
                            offset.append(0)
                        else:
                            lowest_gridpoint = max(
                                interval.start.offset, iteration_interval.start.offset
                            )
                            off = acc.offset[dim] + (
                                lowest_gridpoint - iteration_interval.start.offset
                            )
                            offset.append(min(0, off))
                    else:
                        offset.append(0)
                offset.append(acc.offset[2])
            else:
                offset = [min(acc.offset[0], 0), min(acc.offset[1], 0), acc.offset[2]]

            if offset[2] is None:
                offset[2] = 0

            origins.setdefault(acc.field, offset)
            origins[acc.field] = (
                min(origins[acc.field][0], offset[0]),
                min(origins[acc.field][1], offset[1]),
                min(origins[acc.field][2], offset[2]),
            )

        for name, origin in origins.items():
            origins[name] = (
                -origin[0] - self.node.iteration_space.i_interval.start.offset,
                -origin[1] - self.node.iteration_space.j_interval.start.offset,
                -origin[2],
            )
        return origins

    def get_innermost_memlets(self):

        access_collection: AccessCollector.CartesianAccessCollection = get_access_collection(
            self.node
        )
        region_accesses = {
            acc.field for acc in access_collection.ordered_accesses() if acc.region is not None
        }
        variable_k_accesses = {
            acc.name
            for acc in self.node.oir_node.iter_tree().if_isinstance(oir.FieldAccess)
            if isinstance(acc.offset, common.VariableKOffset)
        }
        dynamic_accesses = {
            get_tasklet_symbol(acc.name, acc.offset.to_tuple(), is_target=False)
            for maskstmt in self.node.oir_node.iter_tree().if_isinstance(oir.MaskStmt)
            for stmt in maskstmt.body
            for assign in stmt.iter_tree().if_isinstance(oir.AssignStmt)
            for acc in assign.right.iter_tree().if_isinstance(oir.FieldAccess)
        }
        dynamic_accesses |= {
            get_tasklet_symbol(acc.name, acc.offset.to_tuple(), is_target=True)
            for maskstmt in self.node.oir_node.iter_tree().if_isinstance(oir.MaskStmt)
            for stmt in maskstmt.body
            for assign in stmt.iter_tree().if_isinstance(oir.AssignStmt)
            for acc in assign.left.iter_tree().if_isinstance(oir.FieldAccess)
        }

        dynamic_accesses |= {
            get_tasklet_symbol(acc.name, acc.offset.to_tuple(), is_target=False)
            for assign in self.node.oir_node.iter_tree().if_isinstance(oir.AssignStmt)
            for acc in assign.right.iter_tree().if_isinstance(oir.FieldAccess)
            if isinstance(acc.offset, common.VariableKOffset)
        }
        dynamic_accesses |= {
            get_tasklet_symbol(acc.name, acc.offset.to_tuple(), is_target=True)
            for assign in self.node.oir_node.iter_tree().if_isinstance(oir.AssignStmt)
            for acc in assign.right.iter_tree().if_isinstance(oir.FieldAccess)
            if isinstance(acc.offset, common.VariableKOffset)
        }
        in_memlets = dict()
        for name, offsets in access_collection.read_offsets().items():
            shape = [
                edge
                for edge in self.parent_state.in_edges(self.node)
                if edge.dst_conn == f"IN_{name}"
            ][0].data.subset.bounding_box_size()
            dynamic_subset_strs = [f"0:{s}" for s in shape]
            dimensions = array_dimensions(self.parent_sdfg.arrays[name])
            data_dims = self.parent_sdfg.arrays[name].shape[sum(dimensions) :]
            origin = self.compensated_origins[name]

            for offset in offsets:
                idx_subset_strs = _get_offset_subset_str(origin, offset, dimensions)
                if name in region_accesses:
                    subset_strs = dynamic_subset_strs[: sum(dimensions[:2])]
                    ij_offs = [0, 0]
                else:
                    subset_strs = idx_subset_strs[: (sum(dimensions[:2]))]
                    ij_offs = list(offset[:2])
                if name in variable_k_accesses:
                    acc_name = get_tasklet_symbol(name, ij_offs + [0], is_target=False)
                    subset_strs.append(dynamic_subset_strs[sum(dimensions[:2])])
                elif dimensions[2]:
                    acc_name = get_tasklet_symbol(name, ij_offs + [offset[2]], is_target=False)
                    subset_strs.append(idx_subset_strs[sum(dimensions[:2])])
                else:
                    acc_name = get_tasklet_symbol(name, ij_offs + [0], is_target=False)
                if data_dims:
                    subset_strs.extend(f"0:{dim}" for dim in data_dims)
                in_memlets[acc_name] = dace.memlet.Memlet.simple(
                    name,
                    ",".join(subset_strs),
                    dynamic=name in region_accesses
                    or name in variable_k_accesses
                    or acc_name in dynamic_accesses,
                )

        out_memlets = dict()
        for name in access_collection.write_fields():
            shape = [
                edge
                for edge in self.parent_state.out_edges(self.node)
                if edge.src_conn == f"OUT_{name}"
            ][0].data.subset.bounding_box_size()
            dynamic_subset_strs = [f"0:{s}" for s in shape]
            dimensions = array_dimensions(self.parent_sdfg.arrays[name])
            data_dims = self.parent_sdfg.arrays[name].shape[sum(dimensions) :]
            idx_subset_strs = _get_offset_subset_str(
                self.compensated_origins[name], (0, 0, 0), dimensions
            )
            idx_subset_strs.extend(f"0:{dim}" for dim in data_dims)
            if name in region_accesses:
                subset_strs = (
                    dynamic_subset_strs[: sum(dimensions[:2])]
                    + idx_subset_strs[sum(dimensions[:2]) :]
                )
            else:
                subset_strs = idx_subset_strs

            acc_name = "__" + name
            out_memlets[acc_name] = dace.memlet.Memlet.simple(
                name,
                ",".join(subset_strs),
                dynamic=acc_name in dynamic_accesses or acc_name in dynamic_accesses,
            )
        return in_memlets, out_memlets

    def add_nodes_and_edges(self):
        in_memlets, out_memlets = self.get_innermost_memlets()
        from collections import OrderedDict

        map_ranges = OrderedDict(
            j=get_interval_range_str(self.node.iteration_space.j_interval, "__J"),
            i=get_interval_range_str(self.node.iteration_space.i_interval, "__I"),
        )
        inputs = [name[len("IN_") :] for name in self.node.in_connectors]
        outputs = [name[len("OUT_") :] for name in self.node.out_connectors]
        input_nodes = {name: self.res_state.add_read(name) for name in inputs}
        output_nodes = {name: self.res_state.add_write(name) for name in outputs}

        dimensions = {
            name: array_dimensions(array) for name, array in self.parent_sdfg.arrays.items()
        }
        nonflat_dimensions = {}
        access_collection = get_access_collection(self.node)
        for name, dims in dimensions.items():
            if name in access_collection.read_offsets():
                offsets = access_collection.read_offsets()[name]
                k_offsets = set((0, 0, o[2]) for o in offsets)
                for offset in k_offsets:

                    acc_name = name + "__" + _get_offset_suffix(offset)
                    if acc_name in in_memlets:
                        in_dims = list(dims)
                        shape = iter(in_memlets[acc_name].subset.bounding_box_size())
                        flat_dims = [
                            i for i, d in enumerate(in_dims) if d and True == (next(shape) == 1)
                        ]
                        for i in flat_dims:
                            in_dims[i] = False
                        nonflat_dimensions[acc_name] = in_dims
            if "__" + name in out_memlets:
                out_dims = list(dims)
                shape = iter(out_memlets["__" + name].subset.bounding_box_size())
                flat_dims = [i for i, d in enumerate(out_dims) if d and True == (next(shape) == 1)]
                for i in flat_dims:
                    out_dims[i] = False
                nonflat_dimensions[f"__{name}"] = out_dims

        access_collection: AccessCollector.Result = get_access_collection(self.node)
        region_fields = {
            acc.field for acc in access_collection.ordered_accesses() if acc.region is not None
        }
        variable_k_fields = {
            acc.field for acc in access_collection.ordered_accesses() if acc.offset[2] is None
        }
        tasklet_code = TaskletCodegen.apply(
            self.node.oir_node,
            context_dimensions=dimensions,
            nonflat_dimensions=nonflat_dimensions,
            origins=self.compensated_origins,
            region_fields=region_fields,
            variable_k_fields=variable_k_fields,
            index_symbols=self.node.index_symbols,
            global_domain_symbols=self.node.global_domain_symbols,
        )
        tasklet, map_entry, map_exit = self.res_state.add_mapped_tasklet(
            self.node.name + "_tasklet",
            map_ranges=map_ranges,
            inputs=in_memlets,
            outputs=out_memlets,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            code=tasklet_code,
            external_edges=True,
            schedule=self.node.map_schedule,
        )
        for edge in self.res_state.in_edges(tasklet):
            if edge.data.data is None:
                continue
            array = self.res_sdfg.arrays[edge.data.data]
            if any(
                isinstance(acc.offset, common.VariableKOffset)
                for acc in self.node.oir_node.iter_tree().if_isinstance(oir.FieldAccess)
                if edge.data.data == acc.name
            ):
                tasklet.in_connectors[edge.dst_conn] = dace.pointer(array.dtype)


class BlockVerticalLoopExpander(NaiveVerticalLoopExpander):

    default_tile_sizes = (64, 8)

    def get_tiled_subset_strs(self, nsdfg, iteration_space):
        tile_sizes = self.node.tile_sizes or self.default_tile_sizes

        access_collection = get_access_collection(nsdfg.sdfg)
        region_fields = {
            acc.field for acc in access_collection.ordered_accesses() if acc.region is not None
        }

        # collect inner subsets
        inner_subsets: Dict[str, dace.subsets.Subset] = dict()
        for state in nsdfg.sdfg.states():
            for edge in state.edges():
                if edge.data.data is not None:
                    name = edge.data.data
                    array: dace.data.Array = nsdfg.sdfg.arrays[name]
                    dims = array_dimensions(array)

                    ij_subset = dace.subsets.Range(edge.data.subset.ranges[: sum(dims[:2])])
                    subset_union = inner_subsets.get(name, ij_subset)
                    inner_subsets[name] = dace.subsets.union(subset_union, ij_subset)

        # confine subset to tile
        for name, subset in inner_subsets.items():
            if name in region_fields:
                continue
            array: dace.data.Array = nsdfg.sdfg.arrays[name]
            dims = array_dimensions(array)
            if dims[0]:
                irange = list(subset.ranges[0])
                irange[0] += dace.symbol("tile_i")
                irange[1] = dace.symbolic.pystr_to_symbolic(irange[1]).replace(
                    dace.symbol("__I"),
                    dace.symbolic.pystr_to_symbolic(f"min(tile_i+({tile_sizes[0]}),__I)"),
                )
                subset.ranges[0] = tuple(irange)
            if dims[1]:
                j_idx = 1 if dims[0] else 0
                jrange = list(subset.ranges[j_idx])
                jrange[0] += dace.symbol("tile_j")
                jrange[1] = dace.symbolic.pystr_to_symbolic(jrange[1]).replace(
                    dace.symbol("__J"),
                    dace.symbolic.pystr_to_symbolic(f"min(tile_j+({tile_sizes[1]}),__J)"),
                )
                subset.ranges[1] = tuple(jrange)
        for name, subset in inner_subsets.items():

            array: dace.data.Array = nsdfg.sdfg.arrays[name]
            dims = array_dimensions(array)
            inner_subsets[name] = dace.subsets.Range(
                subset.ranges + [(0, s - 1, 1) for s in array.shape[sum(dims[:2]) :]]
            )
        return {k: str(v) for k, v in inner_subsets.items()}

    def device_map(self, nsdfg):

        tile_i_sym = dace.symbol("tile_i", dtype=dace.int32)
        tile_j_sym = dace.symbol("tile_j", dtype=dace.int32)
        global_I_sym = dace.symbol("__global_I", dtype=dace.int32)
        global_J_sym = dace.symbol("__global_J", dtype=dace.int32)

        self.res_state.add_node(nsdfg)

        nsdfg.sdfg.parent = self.res_state
        nsdfg.sdfg.parent_sdfg = self.res_state.parent
        nsdfg.sdfg.parent_nsdfg_node = nsdfg

        iteration_space_bounding_box = CartesianIterationSpace.domain()
        for _, section in self.node.sections:
            for node, _ in section.all_nodes_recursive():
                if isinstance(node, HorizontalExecutionLibraryNode):
                    iteration_space_bounding_box = (
                        iteration_space_bounding_box & node.iteration_space
                    )
        i_range = get_interval_range_str(iteration_space_bounding_box.i_interval, "__I")
        j_range = get_interval_range_str(iteration_space_bounding_box.j_interval, "__J")
        tile_sizes = self.node.tile_sizes or self.default_tile_sizes
        from collections import OrderedDict

        ndrange = OrderedDict(
            tile_j=f"{j_range}:{tile_sizes[1]}",
            tile_i=f"{i_range}:{tile_sizes[0]}",
        )

        map_entry, map_exit = self.res_state.add_map(
            self.node.name + "_device_map", ndrange, schedule=self.node.tiling_map_schedule
        )
        subset_strs = self.get_tiled_subset_strs(nsdfg, iteration_space_bounding_box)
        if not nsdfg.in_connectors:
            self.res_state.add_edge(map_entry, None, nsdfg, None, dace.Memlet())
        for name in nsdfg.in_connectors:
            subset_all = ",".join(f"0:{s}" for s in self.res_sdfg.arrays[name].shape)
            read_node = self.res_state.add_read(name)
            self.res_state.add_edge_pair(
                scope_node=map_entry,
                external_node=read_node,
                internal_node=nsdfg,
                internal_connector=name,
                internal_memlet=dace.Memlet.simple(name, subset_strs[name]),
                external_memlet=dace.Memlet.simple(name, subset_all),
                scope_connector=name,
            )
        if not nsdfg.out_connectors:
            self.res_state.add_edge(nsdfg, None, map_exit, None, dace.Memlet())
        for name in nsdfg.out_connectors:
            subset_all = ",".join(f"0:{s}" for s in self.res_sdfg.arrays[name].shape)
            write_node = self.res_state.add_write(name)
            self.res_state.add_edge_pair(
                scope_node=map_exit,
                external_node=write_node,
                internal_node=nsdfg,
                internal_connector=name,
                internal_memlet=dace.Memlet.simple(name, subset_strs[name]),
                external_memlet=dace.Memlet.simple(name, subset_all),
                scope_connector=name,
            )
        nsdfg.symbol_mapping["__I"] = dace.symbolic.pystr_to_symbolic(
            f"min({tile_sizes[0]}, __I - tile_i )"
        )
        nsdfg.symbol_mapping["__J"] = dace.symbolic.pystr_to_symbolic(
            f"min({tile_sizes[1]}, __J - tile_j )"
        )

        state: dace.SDFGState
        access_collection: AccessCollector.Result = get_access_collection(self.node)
        region_fields = {
            acc.field for acc in access_collection.ordered_accesses() if acc.region is not None
        }
        for edge, state in nsdfg.sdfg.all_edges_recursive():
            if (
                isinstance(edge.data, dace.InterstateEdge)
                or edge.data.data is None
                or edge.data.data not in region_fields
            ):
                continue

            is_he_edge = isinstance(edge.dst, HorizontalExecutionLibraryNode) or isinstance(
                edge.src, HorizontalExecutionLibraryNode
            )
            if is_he_edge:
                if isinstance(edge.dst, HorizontalExecutionLibraryNode):
                    node = edge.dst
                else:
                    node = edge.src
                he_access_collection: AccessCollector.Result = get_access_collection(node)
                he_region_fields = {
                    acc.field
                    for acc in he_access_collection.ordered_accesses()
                    if acc.region is not None
                }
            else:
                he_region_fields = set()
            if is_he_edge and edge.data.data not in he_region_fields:
                dims = array_dimensions(self.res_sdfg.arrays[edge.data.data])
                if dims[0]:
                    irange = list(edge.data.subset.ranges[0])
                    irange[0] += tile_i_sym
                    irange[1] += tile_i_sym
                    edge.data.subset.ranges[0] = tuple(irange)
                if dims[1]:
                    jrange = list(edge.data.subset.ranges[1 if dims[0] else 0])
                    jrange[0] += tile_j_sym
                    jrange[1] += tile_j_sym
                    edge.data.subset.ranges[1 if dims[0] else 0] = tuple(jrange)
            else:
                edge.data.subset.replace(
                    {
                        dace.symbol("__I"): global_I_sym,
                        dace.symbol("__J"): global_J_sym,
                    }
                )

                array = state.parent.arrays[edge.data.data]
                shape = list(dace.symbolic.pystr_to_symbolic(s) for s in array.shape)
                for i, s in enumerate(shape):
                    s = s.replace(dace.symbol("__I"), global_I_sym)
                    s = s.replace(dace.symbol("__J"), global_J_sym)
                    shape[i] = s
                array.shape = tuple(shape)

        if "tile_i" not in nsdfg.sdfg.symbols:
            nsdfg.sdfg.add_symbol("tile_i", dace.int32)
        if "tile_j" not in nsdfg.sdfg.symbols:
            nsdfg.sdfg.add_symbol("tile_j", dace.int32)
        if "__global_I" not in nsdfg.sdfg.symbols:
            nsdfg.sdfg.add_symbol("__global_I", dace.int32)
        if "__global_J" not in nsdfg.sdfg.symbols:
            nsdfg.sdfg.add_symbol("__global_J", dace.int32)

        for node, _ in nsdfg.sdfg.all_nodes_recursive():
            if isinstance(node, HorizontalExecutionLibraryNode):
                node.index_symbols = ["(tile_i+i)", "(tile_j+j)", "0"]
                node.global_domain_symbols = ["__global_I", "__global_J"]
            elif isinstance(node, dace.nodes.NestedSDFG):
                node.symbol_mapping["tile_i"] = tile_i_sym
                node.symbol_mapping["tile_j"] = tile_j_sym
                node.symbol_mapping["__global_I"] = global_I_sym
                node.symbol_mapping["__global_J"] = global_J_sym
                if "tile_i" not in node.sdfg.symbols:
                    node.sdfg.add_symbol("tile_i", dace.int32)
                if "tile_j" not in node.sdfg.symbols:
                    node.sdfg.add_symbol("tile_j", dace.int32)
                if "__global_I" not in node.sdfg.symbols:
                    node.sdfg.add_symbol("__global_I", dace.int32)
                if "__global_J" not in node.sdfg.symbols:
                    node.sdfg.add_symbol("__global_J", dace.int32)

        nsdfg.symbol_mapping["__global_I"] = dace.symbolic.pystr_to_symbolic("__I")
        nsdfg.symbol_mapping["__global_J"] = dace.symbolic.pystr_to_symbolic("__J")
        nsdfg.symbol_mapping["tile_i"] = tile_i_sym
        nsdfg.symbol_mapping["tile_j"] = tile_j_sym


class SequentialBlockLoopExpander(BlockVerticalLoopExpander):
    def add_nodes_and_edges(self):
        inner_expander = SequentialNaiveVerticalLoopExpander(
            self.node, self.parent_state, self.parent_sdfg, fix_context_memlets=False
        )
        res_nsdfg = inner_expander.expand()
        self.device_map(res_nsdfg)


class ParallelBlockLoopExpander(BlockVerticalLoopExpander):
    def add_nodes_and_edges(self):
        res = ParallelNaiveVerticalLoopExpander(
            self.node, self.parent_state, self.parent_sdfg, fix_context_memlets=False
        ).expand()
        self.device_map(res)


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


@dace.library.register_expansion(VerticalLoopLibraryNode, "block")
class BlockVerticalLoopExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def expansion(
        node: "VerticalLoopLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.nodes.NestedSDFG:
        from gtc.common import LoopOrder

        if node.loop_order == LoopOrder.PARALLEL:
            return ParallelBlockLoopExpander(node, parent_state, parent_sdfg).expand()
        else:
            return SequentialBlockLoopExpander(node, parent_state, parent_sdfg).expand()


@dace.library.register_expansion(HorizontalExecutionLibraryNode, "naive")
class NaiveHorizontalExecutionExpansion(dace.library.ExpandTransformation):
    environments: List = []

    @staticmethod
    def expansion(
        node: "HorizontalExecutionLibraryNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG
    ) -> dace.SDFG:
        return NaiveHorizontalExecutionExpander(node, parent_state, parent_sdfg).expand()
