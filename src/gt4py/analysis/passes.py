# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Definitions and utilities used by all the analysis pipeline components.
"""

import itertools
import warnings
from typing import Optional, Set, Tuple, Union

from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py.analysis import (
    DomainBlockInfo,
    IJBlockInfo,
    IntervalBlockInfo,
    IntervalInfo,
    StatementInfo,
    SymbolInfo,
    TransformData,
    TransformPass,
)
from gt4py.definitions import Extent


class IRSpecificationError(gt_definitions.GTSpecificationError):
    def __init__(self, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = "Invalid specification"
            else:
                message = (
                    f"Invalid specification in '{loc.scope}' (line: {loc.line}, col: {loc.col})"
                )
        super().__init__(message, loc=loc)


class IntervalSpecificationError(IRSpecificationError):
    def __init__(self, interval, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid interval specification '{interval}' "
            else:
                message = f"Invalid interval specification '{interval}' in '{loc.scope}' (line: {loc.line}, col: {loc.col})"
        super().__init__(message, loc=loc)


class DataTypeSpecificationError(IRSpecificationError):
    def __init__(self, data_type, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid data type specification '{data_type}'"
            else:
                message = f"Invalid data type specification '{data_type}' in '{loc.scope}' (line: {loc.line}, col: {loc.col})"
        super().__init__(message, loc=loc)


class InitInfoPass(TransformPass):

    _DEFAULT_OPTIONS = {"redundant_temp_fields": False}

    class SymbolMaker(gt_ir.IRNodeVisitor):
        def __init__(self, transform_data: TransformData, redundant_temp_fields: bool):
            self.data = transform_data
            self.redundant_temp_fields = redundant_temp_fields

        def visit_Ref(self, node: gt_ir.Ref):
            if node.name not in self.data.symbols:
                raise IRSpecificationError("Reference to undefined symbol", loc=node.loc)

        def visit_Decl(self, node: gt_ir.FieldDecl):
            self._add_symbol(node)
            return None

        def visit_BlockStmt(self, node: gt_ir.BlockStmt):
            result = [self.visit(stmt) for stmt in node.stmts]
            return [item for item in result if item is not None]

        def visit_StencilDefinition(self, node: gt_ir.StencilDefinition):
            # Add API symbols first
            for decl in itertools.chain(node.api_fields, node.parameters):
                self._add_symbol(decl)

            # Build the information tables
            for computation in node.computations:
                self.visit(computation)

        def _add_symbol(self, decl):
            has_redundancy = (
                isinstance(decl, gt_ir.FieldDecl)
                and self.redundant_temp_fields
                and not decl.is_api
            )
            symbol_info = SymbolInfo(decl, has_redundancy=has_redundancy)
            self.data.symbols[decl.name] = symbol_info

    class IntervalMaker(gt_ir.IRNodeVisitor):
        def __init__(self, transform_data: TransformData):
            self.data = transform_data

        def visit_StencilDefinition(self, node: gt_ir.StencilDefinition):
            self.data.splitters_var = None
            self.data.min_k_interval_sizes = [0]

            # First, look for dynamic splitters variable
            for computation in node.computations:
                interval_def = computation.interval
                for axis_bound in [interval_def.start, interval_def.end]:
                    if isinstance(axis_bound.level, gt_ir.VarRef):
                        name = axis_bound.level.name
                        for item in node.parameters:
                            if item.name == name:
                                decl = item
                                break
                        else:
                            decl = None

                        if decl is None or decl.length == 0:
                            raise IntervalSpecificationError(
                                interval_def,
                                "Invalid variable reference in interval specification",
                                loc=axis_bound.loc,
                            )

                        self.data.splitters_var = decl.name
                        self.data.min_k_interval_sizes = [1] * (decl.length + 1)

            # Extract computation intervals
            computation_intervals = []
            for computation in node.computations:
                # Process current interval definition
                interval_def = computation.interval
                bounds = [None, None]

                for i, axis_bound in enumerate([interval_def.start, interval_def.end]):
                    if isinstance(axis_bound.level, gt_ir.VarRef):
                        # Dynamic splitters: check existing reference and extract size info
                        if axis_bound.level.name != self.data.splitters_var:
                            raise IntervalSpecificationError(
                                interval_def,
                                "Non matching variable reference in interval specification",
                                loc=axis_bound.loc,
                            )

                        index = axis_bound.level.index + 1
                        offset = axis_bound.offset
                        if offset < 0:
                            index = index - 1

                    else:
                        # Static splitter: extract size info
                        index = (
                            self.data.nk_intervals
                            if axis_bound.offset < 0 or axis_bound.level == gt_ir.LevelMarker.END
                            else 0
                        )
                        offset = axis_bound.offset

                        if offset < 0 and axis_bound.level != gt_ir.LevelMarker.END:
                            raise IntervalSpecificationError(
                                interval_def,
                                "Invalid offset in interval specification",
                                loc=axis_bound.loc,
                            )

                        elif offset > 0 and axis_bound.level != gt_ir.LevelMarker.START:
                            raise IntervalSpecificationError(
                                interval_def,
                                "Invalid offset in interval specification",
                                loc=axis_bound.loc,
                            )

                    # Update min sizes
                    if not 0 <= index <= self.data.nk_intervals:
                        raise IntervalSpecificationError(
                            interval_def,
                            "Invalid variable reference in interval specification",
                            loc=axis_bound.loc,
                        )

                    bounds[i] = (index, offset)
                    if index < self.data.nk_intervals:
                        self.data.min_k_interval_sizes[index] = max(
                            self.data.min_k_interval_sizes[index], offset
                        )

                if bounds[0][0] == bounds[1][0] - 1:
                    index = bounds[0][0]
                    min_size = 1 + bounds[0][1] - bounds[1][1]
                    self.data.min_k_interval_sizes[index] = max(
                        self.data.min_k_interval_sizes[index], min_size
                    )

                # Create computation intervals
                interval_info = IntervalInfo(*bounds)
                computation_intervals.append(interval_info)

            return computation_intervals

    class BlockMaker(gt_ir.IRNodeVisitor):
        def __init__(self, transform_data: TransformData, computation_intervals: list):
            self.data = transform_data
            self.computation_intervals = computation_intervals
            self.current_block_info = None
            self.zero_extent = Extent.zeros(transform_data.ndims)

        def visit_Expr(self, node: gt_ir.Expr):
            return []

        def visit_VarRef(self, node: gt_ir.VarRef):
            result = [(node.name, None)]
            return result

        def visit_FieldRef(self, node: gt_ir.FieldRef):
            extent = Extent.from_offset([node.offset.get(ax, 0) for ax in self.data.axes_names])
            result = [(node.name, extent)]
            return result

        def visit_UnaryOpExpr(self, node: gt_ir.UnaryOpExpr):
            result = self.visit(node.arg)
            return result

        def visit_BinOpExpr(self, node: gt_ir.BinOpExpr):
            result = self.visit(node.lhs) + self.visit(node.rhs)
            return result

        def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr):
            result = (
                self.visit(node.condition)
                + self.visit(node.then_expr)
                + self.visit(node.else_expr)
            )
            return result

        def visit_Statement(self, node: gt_ir.Statement):
            assert False

        def visit_Decl(self, node: gt_ir.Decl):
            assert node.is_api is False
            assert node.name in self.data.symbols
            return None

        def visit_Assign(self, node: gt_ir.Assign):
            target_name = node.target.name
            assert target_name in self.data.symbols
            inputs = self._merge_extents(self.visit(node.value))
            result = StatementInfo(self.data.id_generator.new, node, inputs, {target_name})

            return result

        def visit_NativeFuncCall(self, node: gt_ir.NativeFuncCall):
            return [extent for arg in node.args for extent in self.visit(arg)]

        def visit_If(self, node: gt_ir.If):
            inputs = {}
            outputs = set()

            stmts = list(node.main_body.stmts)
            if node.else_body is not None:
                stmts.extend(node.else_body.stmts)
            for stmt in stmts:
                stmt_info = self.visit(stmt)
                inputs = self._merge_extents(list(inputs.items()) + list(stmt_info.inputs.items()))
                outputs |= stmt_info.outputs
            cond_info = self.visit(node.condition)
            inputs = self._merge_extents(list(inputs.items()) + cond_info)

            result = StatementInfo(self.data.id_generator.new, node, inputs, outputs)

            return result

        def visit_BlockStmt(self, node: gt_ir.BlockStmt):
            inputs = {}
            outputs = set()
            for stmt in node.stmts:
                stmt_info = self.visit(stmt)
                inputs = self._merge_extents(list(inputs.items()) + list(stmt_info.inputs.items()))
                outputs |= stmt_info.outputs

            result = StatementInfo(self.data.id_generator.new, node, inputs, outputs)

            return result

        def visit_ComputationBlock(self, node: gt_ir.ComputationBlock):
            interval = next(iter(self.current_block_info.intervals))
            interval_block = IntervalBlockInfo(self.data.id_generator.new, interval)

            assert node.body.stmts  # non-empty computation
            stmt_infos = [
                info for info in [self.visit(stmt) for stmt in node.body.stmts] if info is not None
            ]
            group_outputs = set()

            # Traverse computation statements
            for stmt_info in stmt_infos:
                stmt_inputs_with_ij_offset = set(
                    [
                        input
                        for input, extent in stmt_info.inputs.items()
                        if extent[:2] != ((0, 0), (0, 0))
                    ]
                )

                # Open a new stage when it is not possible to use the current one
                if not group_outputs.isdisjoint(stmt_inputs_with_ij_offset):
                    assert interval_block.stmts
                    assert interval_block.outputs
                    # If some output field is read with an offset it likely implies different
                    # compute extent
                    self.current_block_info.ij_blocks.append(
                        self._make_ij_block(interval, interval_block)
                    )
                    interval_block = IntervalBlockInfo(self.data.id_generator.new, interval)
                    group_outputs = set()

                interval_block.stmts.append(stmt_info)
                interval_block.outputs |= stmt_info.outputs
                for name, extent in stmt_info.inputs.items():
                    interval_block.inputs[name] = interval_block.inputs.get(name, extent) | extent

                group_outputs |= stmt_info.outputs

            if interval_block.stmts:
                self.current_block_info.ij_blocks.append(
                    self._make_ij_block(interval, interval_block)
                )

        def visit_StencilDefinition(self, node: gt_ir.StencilDefinition):
            for computation, interval in zip(node.computations, self.computation_intervals):
                self.current_block_info = DomainBlockInfo(
                    self.data.id_generator.new, computation.iteration_order, {interval}, []
                )
                self.visit(computation)
                self.data.blocks.append(self.current_block_info)

        def _make_ij_block(self, interval, interval_block):
            ij_block = IJBlockInfo(
                self.data.id_generator.new,
                {interval},
                interval_blocks=[interval_block],
                inputs={**interval_block.inputs},
                outputs=set(interval_block.outputs),
                compute_extent=self.zero_extent,
            )

            return ij_block

        def _merge_extents(self, refs: list):
            result = {}
            params = set()

            # Merge offsets for same symbol
            for name, extent in refs:
                if extent is None:
                    assert name in params or name not in result
                    params |= {name}
                    result.setdefault(name, Extent((0, 0), (0, 0), (0, 0)))
                else:
                    assert name not in params
                    if name in result:
                        result[name] |= extent
                    else:
                        result[name] = extent

            return result

    def __init__(self, redundant_temp_fields=_DEFAULT_OPTIONS["redundant_temp_fields"]):
        self.redundant_temp_fields = redundant_temp_fields

    @property
    def defaults(self):
        return self._DEFAULT_OPTIONS

    def apply(self, transform_data: TransformData):
        interval_maker = self.IntervalMaker(transform_data)
        computation_intervals = interval_maker.visit(transform_data.definition_ir)
        symbol_maker = self.SymbolMaker(transform_data, self.redundant_temp_fields)
        symbol_maker.visit(transform_data.definition_ir)
        block_maker = self.BlockMaker(transform_data, computation_intervals)
        block_maker.visit(transform_data.definition_ir)

        return transform_data


class NormalizeBlocksPass(TransformPass):

    _DEFAULT_OPTIONS = {}

    def __init__(self):
        pass

    @property
    def defaults(self):
        return self._DEFAULT_OPTIONS

    def apply(self, transform_data: TransformData):
        zero_extent = Extent.zeros(transform_data.ndims)
        blocks = []
        for block in transform_data.blocks:
            # Put every statement in a single stage
            for ij_block in block.ij_blocks:
                for interval_block in ij_block.interval_blocks:
                    for stmt_info in interval_block.stmts:
                        interval = interval_block.interval
                        new_interval_block = IntervalBlockInfo(
                            transform_data.id_generator.new,
                            interval,
                            [stmt_info],
                            stmt_info.inputs,
                            stmt_info.outputs,
                        )
                        new_ij_block = IJBlockInfo(
                            transform_data.id_generator.new,
                            {interval},
                            [new_interval_block],
                            {**new_interval_block.inputs},
                            set(new_interval_block.outputs),
                            compute_extent=zero_extent,
                        )
                        new_block = DomainBlockInfo(
                            transform_data.id_generator.new,
                            block.iteration_order,
                            set(new_ij_block.intervals),
                            [new_ij_block],
                            {**new_ij_block.inputs},
                            set(new_ij_block.outputs),
                        )
                        blocks.append(new_block)

        transform_data.blocks = blocks

        return transform_data


class MergeBlocksPass(TransformPass):

    _DEFAULT_OPTIONS = {}

    def __init__(self):
        pass

    @property
    def defaults(self):
        return self._DEFAULT_OPTIONS

    def apply(self, transform_data: TransformData):
        if not transform_data.blocks:  # do not apply pass if empty stencil
            return

        # Greedy strategy to merge multi-stages
        merged_blocks = [transform_data.blocks[0]]
        for candidate in transform_data.blocks[1:]:
            merged = merged_blocks[-1]
            if self._are_compatible_multi_stages(
                merged, candidate, transform_data.has_sequential_axis
            ):
                merged.id = transform_data.id_generator.new
                self._merge_domain_blocks(merged, candidate)
            else:
                merged_blocks.append(candidate)

        # Greedy strategy to merge stages
        # assert transform_data.has_sequential_axis
        for block in merged_blocks:
            merged_ijs = [block.ij_blocks[0]]
            for ij_candidate in block.ij_blocks[1:]:
                merged = merged_ijs[-1]
                if self._are_compatible_stages(
                    merged,
                    ij_candidate,
                    transform_data.min_k_interval_sizes,
                    block.iteration_order,
                ):
                    merged.id = transform_data.id_generator.new
                    self._merge_ij_blocks(merged, ij_candidate, transform_data)
                else:
                    merged_ijs.append(ij_candidate)

            block.ij_blocks = merged_ijs

        transform_data.blocks = merged_blocks

        return transform_data

    def _are_compatible_multi_stages(
        self, target: DomainBlockInfo, candidate: DomainBlockInfo, has_sequential_axis: bool
    ):
        result = False
        if candidate.iteration_order == target.iteration_order:
            if candidate.iteration_order == gt_ir.IterationOrder.PARALLEL and has_sequential_axis:
                inputs_with_k_deps = set(
                    name for name, extent in candidate.inputs.items() if extent[-1] != (0, 0)
                )
                result = target.outputs.isdisjoint(inputs_with_k_deps)
            else:
                result = True

        return result

    def _are_compatible_stages(
        self,
        target: IJBlockInfo,
        candidate: IJBlockInfo,
        min_k_interval_sizes: list,
        iteration_order: gt_ir.IterationOrder,
    ):
        # Check that the two stages have the same compute extent
        if not (target.compute_extent == candidate.compute_extent):
            return False

        result = True
        # Check that there is not overlap between stage intervals and that
        # merging stages will not imply a reordering of the execution order
        for interval in target.intervals:
            for candidate_interval in candidate.intervals:
                if (
                    interval != candidate_interval
                    and interval.overlaps(candidate_interval, min_k_interval_sizes)
                ) or interval.precedes(candidate_interval, min_k_interval_sizes, iteration_order):
                    result = False

        # Check that there are not data dependencies between stages
        if result:
            for input, extent in candidate.inputs.items():
                if result:
                    if input in target.outputs:
                        read_interval = (
                            next(iter(candidate.intervals)).as_tuple(min_k_interval_sizes)
                            + extent[-2]
                        )
                        for merged_interval_block in target.interval_blocks:
                            merged_interval = merged_interval_block.interval
                            if merged_interval.as_tuple(
                                min_k_interval_sizes
                            ) != read_interval and merged_interval.overlaps(
                                read_interval, min_k_interval_sizes
                            ):
                                result = False
                                break
        return result

    def _merge_domain_blocks(self, target, candidate):
        target.ij_blocks.extend(candidate.ij_blocks)
        target.intervals |= candidate.intervals
        target.outputs |= candidate.outputs
        for name, extent in candidate.inputs.items():
            if name in target.inputs:
                target.inputs[name] |= extent
            else:
                target.inputs[name] = extent

    def _merge_ij_blocks(self, target, candidate, transform_data):
        target_intervals = {
            target_int_block.interval: target_int_block
            for target_int_block in target.interval_blocks
        }
        for candidate_int_block in candidate.interval_blocks:
            if candidate_int_block.interval in target_intervals:
                merged_int_block = target_intervals[candidate_int_block.interval]
                merged_int_block.id = transform_data.id_generator.new
                merged_int_block.stmts.extend(candidate_int_block.stmts)

                for name, extent in candidate_int_block.inputs.items():
                    if name in merged_int_block.inputs:
                        merged_int_block.inputs[name] |= extent
                    else:
                        merged_int_block.inputs[name] = extent

                merged_int_block.outputs |= candidate_int_block.outputs

            else:
                target.interval_blocks.append(candidate_int_block)

        target.intervals |= candidate.intervals
        target.outputs |= candidate.outputs
        for name, extent in candidate.inputs.items():
            if name in target.inputs:
                target.inputs[name] |= extent
            else:
                target.inputs[name] = extent


class ComputeExtentsPass(TransformPass):

    _DEFAULT_OPTIONS = {}

    def __init__(self):
        pass

    @property
    def defaults(self):
        return self._DEFAULT_OPTIONS

    def apply(self, transform_data: TransformData):
        seq_axis = transform_data.definition_ir.domain.index(
            transform_data.definition_ir.domain.sequential_axis
        )
        access_extents = {}
        for name in transform_data.symbols:
            access_extents[name] = Extent.zeros()

        blocks = transform_data.blocks
        for block in reversed(blocks):
            for ij_block in reversed(block.ij_blocks):
                ij_block.compute_extent = Extent.zeros()
                for name in ij_block.outputs:
                    ij_block.compute_extent |= access_extents[name]
                for int_block in ij_block.interval_blocks:
                    for name, extent in int_block.inputs.items():
                        extent = Extent(
                            list(extent[:seq_axis]) + [(0, 0)]
                        )  # exclude sequential axis
                        accumulated_extent = ij_block.compute_extent + extent
                        access_extents[name] |= accumulated_extent

        transform_data.implementation_ir.fields_extents = {
            name: Extent(extent) for name, extent in access_extents.items()
        }

        return transform_data


class DataTypePass(TransformPass):
    class CollectDataTypes(gt_ir.IRNodeVisitor):
        def __call__(self, node):
            assert isinstance(node, gt_ir.StencilImplementation)
            self.vars = node.parameters
            self.fields = node.fields
            self.visit(node)

        def visit_ApplyBlock(self, node: gt_ir.Node, **kwargs):
            self.generic_visit(node, apply_block_symbols=node.local_symbols, **kwargs)

        def visit_Assign(self, node: gt_ir.Assign, **kwargs):
            self.visit(node.value, **kwargs)
            if hasattr(node.target, "data_type") and node.target.data_type != node.value.data_type:
                raise Exception(
                    "Symbol '{}' used with inconsistent data types.".format(node.target.name)
                )
            node.target.data_type = getattr(node.value, "data_type", gt_ir.DataType.AUTO)
            if self.fields[node.target.name].data_type == gt_ir.DataType.AUTO:
                self.fields[node.target.name].data_type = node.value.data_type
            self.visit(node.target, **kwargs)

        def visit_VarRef(self, node: gt_ir.Node, apply_block_symbols={}, **kwargs):
            self.generic_visit(node, **kwargs)

            if node.name in apply_block_symbols:
                var_decl = apply_block_symbols[node.name]
            else:
                var_decl = self.vars[node.name]

            if var_decl.data_type == gt_ir.DataType.AUTO:
                var_decl.data_type = node.data_type
            else:
                node.data_type = var_decl.data_type

        def visit_FieldRef(self, node: gt_ir.Node, **kwargs):
            self.generic_visit(node, **kwargs)
            if self.fields[node.name].data_type == gt_ir.DataType.AUTO:
                self.fields[node.name].data_type = node.data_type
            else:
                node.data_type = self.fields[node.name].data_type

        def visit_UnaryOpExpr(self, node: gt_ir.Node, **kwargs):
            self.generic_visit(node, **kwargs)
            assert node.arg.data_type is not gt_ir.DataType.AUTO
            if node.op.value in [gt_ir.UnaryOperator.NOT]:
                node.data_type = gt_ir.DataType.from_dtype(bool)
            else:
                node.data_type = node.arg.data_type

        def visit_BinOpExpr(self, node: gt_ir.Node, **kwargs):
            self.generic_visit(node, **kwargs)
            assert node.lhs.data_type is not gt_ir.DataType.AUTO
            assert node.rhs.data_type is not gt_ir.DataType.AUTO
            if node.op.value in [
                gt_ir.BinaryOperator.OR,
                gt_ir.BinaryOperator.EQ,
                gt_ir.BinaryOperator.NE,
                gt_ir.BinaryOperator.LT,
                gt_ir.BinaryOperator.LE,
                gt_ir.BinaryOperator.GT,
                gt_ir.BinaryOperator.GE,
            ]:
                node.data_type = gt_ir.DataType.from_dtype(bool)
            else:
                node.data_type = gt_ir.DataType.merge(node.lhs.data_type, node.rhs.data_type)

        def visit_TernaryOpExpr(self, node: gt_ir.TernaryOpExpr, **kwargs):
            self.generic_visit(node, **kwargs)
            assert node.then_expr.data_type is not gt_ir.DataType.AUTO
            assert node.else_expr.data_type is not gt_ir.DataType.AUTO
            assert node.condition.data_type is not gt_ir.DataType.AUTO
            node.data_type = gt_ir.DataType.merge(
                node.then_expr.data_type, node.else_expr.data_type
            )

        def visit_NativeFuncCall(self, node: gt_ir.NativeFuncCall, **kwargs):
            self.generic_visit(node.args, **kwargs)

            data_type = set(
                arg.data_type for arg in node.args if arg.data_type != gt_ir.DataType.DEFAULT
            )
            if len(data_type) > 1:
                raise DataTypeSpecificationError(
                    "Builtin function call with mixed data types", loc=node.loc
                )
            data_type = data_type.pop() if len(data_type) else gt_ir.DataType.DEFAULT

            if node.func in (
                gt_ir.NativeFunction.MIN,
                gt_ir.NativeFunction.MAX,
                gt_ir.NativeFunction.ABS,
                gt_ir.NativeFunction.MOD,
            ):
                node.data_type = data_type
            elif node.func in (
                gt_ir.NativeFunction.ISFINITE,
                gt_ir.NativeFunction.ISINF,
                gt_ir.NativeFunction.ISNAN,
            ):
                node.data_type = gt_ir.DataType.BOOL
            elif node.func in (
                gt_ir.NativeFunction.CEIL,
                gt_ir.NativeFunction.FLOOR,
                gt_ir.NativeFunction.TRUNC,
            ):
                node.data_type = gt_ir.DataType.INT64
            else:
                node.data_type = gt_ir.DataType.DEFAULT

    def apply(self, transform_data: TransformData):
        collect_data_type = self.CollectDataTypes()
        collect_data_type(transform_data.implementation_ir)
        return transform_data


class ComputeUsedSymbolsPass(TransformPass):
    class ComputeUsedVisitor(gt_ir.IRNodeVisitor):
        def __init__(self, transform_data: TransformData):
            self.data = transform_data

        def visit_VarRef(self, node: gt_ir.VarRef, **kwargs):
            self.data.symbols[node.name].in_use = True

        def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs):
            self.data.symbols[node.name].in_use = True

    def apply(self, transform_data: TransformData):
        visitor = self.ComputeUsedVisitor(transform_data)
        visitor.visit(transform_data.definition_ir)
        return transform_data


class BuildIIRPass(TransformPass):

    _DEFAULT_OPTIONS = {}

    def __init__(self):
        self.data = None
        self.iir = None

    @property
    def defaults(self):
        return self._DEFAULT_OPTIONS

    def apply(self, transform_data: TransformData):
        self.data = transform_data
        self.iir = transform_data.implementation_ir

        if self.data.splitters_var:
            self.iir.axis_splitters_var = self.data.splitters_var

        # Signature
        self.iir.api_signature = self.data.definition_ir.api_signature

        # Create fields and parameters
        for name, symbol in self.data.symbols.items():
            if not symbol.in_use:
                self.iir.unreferenced.append(name)

            decl = symbol.decl
            if isinstance(decl, gt_ir.VarDecl):
                self.iir.parameters[name] = decl
            else:
                self.iir.fields[name] = decl

        # Create multistages
        for block in self.data.blocks:
            groups = [
                gt_ir.StageGroup(stages=[self._make_stage(ij_block)])
                for ij_block in block.ij_blocks
            ]
            multi_stage = gt_ir.MultiStage(
                name="multi_stage__{}".format(block.id),
                iteration_order=block.iteration_order,
                groups=groups,
            )
            self.iir.multi_stages.append(multi_stage)

        return transform_data

    def _make_stage(self, ij_block):
        # Apply blocks and decls
        apply_blocks = []
        decls = []
        for int_block in ij_block.interval_blocks:
            # Make apply block
            stmts = []
            local_symbols = {}
            for stmt_info in int_block.stmts:
                if isinstance(stmt_info.stmt, gt_ir.Decl):
                    decl = stmt_info.stmt
                    if decl.name in self.data.symbols:
                        decls.append(stmt_info.stmt)
                    else:
                        assert isinstance(decl, gt_ir.VarDecl)
                        local_symbols[decl.name] = decl
                else:
                    stmts.append(stmt_info.stmt)

            apply_block = gt_ir.ApplyBlock(
                interval=self._make_axis_interval(int_block.interval),
                local_symbols=local_symbols,
                body=gt_ir.BlockStmt(stmts=stmts),
            )
            apply_blocks.append(apply_block)

        # Accessors
        accessors = []
        remaining_outputs = set(ij_block.outputs)
        for name, extent in ij_block.inputs.items():
            if name in remaining_outputs:
                read_write = True
                remaining_outputs.remove(name)
                extent |= Extent.zeros()
            else:
                read_write = False
            accessors.append(self._make_accessor(name, extent, read_write))
        zero_extent = Extent.zeros(self.data.ndims)
        for name in remaining_outputs:
            accessors.append(self._make_accessor(name, zero_extent, True))

        stage = gt_ir.Stage(
            name="stage__{}".format(ij_block.id),
            accessors=accessors,
            apply_blocks=apply_blocks,
            compute_extent=ij_block.compute_extent,
        )

        return stage

    def _make_apply_block(self, interval_block):
        # Body
        stmts = []
        for stmt_info in interval_block.stmts:
            if not isinstance(stmt_info.stmt, gt_ir.Decl):
                stmts.append(stmt_info.stmt)
        body = gt_ir.BlockStmt(stmts=stmts)
        result = gt_ir.ApplyBlock(
            interval=self._make_axis_interval(interval_block.interval), body=body
        )

        return result

    def _make_accessor(self, name, extent, read_write: bool):
        assert name in self.data.symbols
        intent = gt_ir.AccessIntent.READ_WRITE if read_write else gt_ir.AccessIntent.READ_ONLY
        if self.data.symbols[name].is_field:
            assert extent is not None
            result = gt_ir.FieldAccessor(symbol=name, intent=intent, extent=extent)
        else:
            # assert extent is None and not read_write
            assert not read_write
            result = gt_ir.ParameterAccessor(symbol=name)

        return result

    def _make_axis_interval(self, interval: IntervalInfo):
        axis_bounds = []
        for bound in (interval.start, interval.end):
            if bound[0] == 0:
                axis_bounds.append(gt_ir.AxisBound(level=gt_ir.LevelMarker.START, offset=bound[1]))
            elif bound[0] == self.data.nk_intervals:
                axis_bounds.append(gt_ir.AxisBound(level=gt_ir.LevelMarker.END, offset=bound[1]))
            else:
                axis_bounds.append(
                    gt_ir.AxisBound(
                        level=gt_ir.VarRef(name=self.data.splitters_var, index=bound[0] - 1),
                        offset=bound[1],
                    )
                )

        result = gt_ir.AxisInterval(start=axis_bounds[0], end=axis_bounds[1])

        return result


class DemoteLocalTemporariesToVariablesPass(TransformPass):
    class CollectDemotableSymbols(gt_ir.IRNodeVisitor):
        def __call__(self, node: gt_ir.StencilImplementation) -> Set[str]:
            assert isinstance(node, gt_ir.StencilImplementation)
            self.demotables = set(node.temporary_fields)
            self.seen_symbols: Set[str] = set()
            self.stage_symbols: Optional[Set[str]] = None
            self.visit(node)
            return self.demotables

        def visit_Stage(self, node: gt_ir.Stage) -> None:
            self.stage_symbols = set()

            self.generic_visit(node)

            for s in self.stage_symbols:
                if s in self.seen_symbols:
                    self.demotables.discard(s)
                self.seen_symbols.add(s)
            self.stage_symbols = None

        def visit_FieldRef(self, node: gt_ir.FieldRef) -> None:
            if any(v != 0 for v in node.offset.values()):
                self.demotables.discard(node.name)
            assert self.stage_symbols is not None
            self.stage_symbols.add(node.name)

    class DemoteSymbols(gt_ir.IRNodeMapper):
        def __init__(self, demotables):
            self.demotables = demotables
            self.local_symbols = None

        def __call__(self, node: gt_ir.StencilImplementation) -> gt_ir.StencilImplementation:
            assert isinstance(node, gt_ir.StencilImplementation)
            self.fields = node.fields
            node = self.visit(node)
            return node

        def visit_FieldAccessor(
            self, path: tuple, node_name: str, node: gt_ir.FieldAccessor
        ) -> Tuple[bool, Optional[gt_ir.FieldAccessor]]:
            if node.symbol in self.demotables:
                return False, None
            else:
                return True, node

        def visit_StencilImplementation(
            self, path: tuple, node_name: str, node: gt_ir.StencilImplementation
        ) -> gt_ir.StencilImplementation:
            self.iir = node
            res = self.generic_visit(path, node_name, node)
            for f in self.demotables:
                assert f in node.temporary_fields, "Tried to demote api field to variable."
                node.fields.pop(f)
                node.fields_extents.pop(f)
            return res

        def visit_ApplyBlock(
            self, path: tuple, node_name: str, node: gt_ir.ApplyBlock
        ) -> Tuple[bool, gt_ir.ApplyBlock]:
            self.local_symbols = {}

            self.generic_visit(path, node_name, node)

            node.local_symbols.update(self.local_symbols)
            self.local_symbols = None
            return True, node

        def visit_FieldRef(
            self, path: tuple, node_name: str, node: gt_ir.FieldRef
        ) -> Tuple[bool, Union[gt_ir.FieldRef, gt_ir.VarRef]]:
            if node.name in self.demotables:
                if node.name not in self.local_symbols:
                    field_decl = self.fields[node.name]
                    self.local_symbols[node.name] = gt_ir.VarDecl(
                        name=node.name,
                        data_type=field_decl.data_type,
                        length=1,
                        is_api=False,
                        loc=field_decl.loc,
                    )
                return True, gt_ir.VarRef(name=node.name, index=0, loc=node.loc)

            else:
                return True, node

    def apply(self, transform_data: TransformData) -> TransformData:
        collect_demotable_symbols = self.CollectDemotableSymbols()
        demotables = collect_demotable_symbols(transform_data.implementation_ir)

        demote_symbols = self.DemoteSymbols(demotables)
        transform_data.implementation_ir = demote_symbols(transform_data.implementation_ir)

        return transform_data


class HousekeepingPass(TransformPass):
    class WarnIfNoEffect(gt_ir.IRNodeVisitor):
        def __call__(self, stencil_name: str, node: gt_ir.StencilImplementation) -> None:
            assert isinstance(node, gt_ir.StencilImplementation)
            self.stencil_name = stencil_name
            self.visit(node)

        def visit_StencilImplementation(self, node: gt_ir.StencilImplementation):
            # Emit warning if stencil has no effect, i.e. does not read or write to any api fields
            if not node.has_effect:
                warnings.warn(
                    f"Stencil `{self.stencil_name}` has no effect.",
                    RuntimeWarning,
                )

    class PruneEmptyNodes(gt_ir.IRNodeMapper):
        def __call__(self, node: gt_ir.StencilImplementation) -> None:
            assert isinstance(node, gt_ir.StencilImplementation)
            self.visit(node)

        def visit_Stage(
            self, path: tuple, node_name: str, node: gt_ir.Stage
        ) -> Tuple[bool, Optional[gt_ir.Stage]]:
            self.generic_visit(path, node_name, node)

            if any(
                isinstance(a, gt_ir.FieldAccessor) and (a.intent is gt_ir.AccessIntent.READ_WRITE)
                for a in node.accessors
            ):
                return True, node
            else:
                return False, None

        def visit_StageGroup(
            self, path: tuple, node_name: str, node: gt_ir.StageGroup
        ) -> Tuple[bool, Optional[gt_ir.StageGroup]]:
            self.generic_visit(path, node_name, node)

            if node.stages:
                return True, node
            else:
                return False, None

        def visit_MultiStage(
            self, path: tuple, node_name: str, node: gt_ir.MultiStage
        ) -> Tuple[bool, Optional[gt_ir.MultiStage]]:
            self.generic_visit(path, node_name, node)

            if node.groups:
                return True, node
            else:
                return False, None

    def apply(self, transform_data: TransformData) -> TransformData:

        prune_emtpy_nodes = self.PruneEmptyNodes()
        prune_emtpy_nodes(transform_data.implementation_ir)

        self.WarnIfNoEffect()(transform_data.definition_ir.name, transform_data.implementation_ir)

        return transform_data
