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
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

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


MergeableType = TypeVar("MergeableType")
WrappedType = TypeVar("WrappedType")


class IRSpecificationError(gt_definitions.GTSpecificationError):
    def __init__(self, message=None, *, loc=None):
        if message is None:
            message = "Invalid specification"
        message = f"{message} in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message)


class IntervalSpecificationError(IRSpecificationError):
    def __init__(self, interval, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid interval specification '{interval}' "
            else:
                message = f"Invalid interval specification '{interval}' in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)


class DataTypeSpecificationError(IRSpecificationError):
    def __init__(self, data_type, message=None, *, loc=None):
        if message is None:
            if loc is None:
                message = f"Invalid data type specification '{data_type}'"
            else:
                message = f"Invalid data type specification '{data_type}' in '{loc.scope}' (line: {loc.line}, col: {loc.column})"
        super().__init__(message, loc=loc)


class InitInfoPass(TransformPass):
    """Transcribe definition IR structure into blocks.

    Note
    ----
    The following `transform_data` attributes are changed:
        - `symbols`: SymbolInfo for each used symbol.
        - `blocks`: Block structure following the original Definition IR.
    """

    _DEFAULT_OPTIONS = {"redundant_temp_fields": False}

    @staticmethod
    def make_k_intervals(transform_data: TransformData) -> List[IntervalInfo]:
        """Determines intervals over which the computation runs."""
        node: gt_ir.StencilDefinition = transform_data.definition_ir
        transform_data.splitters_var: Optional[str] = None
        transform_data.min_k_interval_sizes: List[int] = [0]

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

                    transform_data.splitters_var = decl.name
                    transform_data.min_k_interval_sizes = [1] * (decl.length + 1)

        # Extract computation intervals
        computation_intervals = []
        for computation in node.computations:
            # Process current interval definition
            interval_def = computation.interval
            bounds = [None, None]

            for i, axis_bound in enumerate([interval_def.start, interval_def.end]):
                if isinstance(axis_bound.level, gt_ir.VarRef):
                    # Dynamic splitters: check existing reference and extract size info
                    if axis_bound.level.name != transform_data.splitters_var:
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
                        transform_data.nk_intervals
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
                if not 0 <= index <= transform_data.nk_intervals:
                    raise IntervalSpecificationError(
                        interval_def,
                        "Invalid variable reference in interval specification",
                        loc=axis_bound.loc,
                    )

                bounds[i] = (index, offset)
                if index < transform_data.nk_intervals:
                    transform_data.min_k_interval_sizes[index] = max(
                        transform_data.min_k_interval_sizes[index], offset
                    )

            if bounds[0][0] == bounds[1][0] - 1:
                index = bounds[0][0]
                min_size = 1 + bounds[0][1] - bounds[1][1]
                transform_data.min_k_interval_sizes[index] = max(
                    transform_data.min_k_interval_sizes[index], min_size
                )

            # Create computation intervals
            interval_info = IntervalInfo(*bounds)
            computation_intervals.append(interval_info)

        return computation_intervals

    class SymbolMaker(gt_ir.IRNodeVisitor):
        """Fills transform_data.symbols with SymbolInfo for each used in the StencilDefinition."""

        @classmethod
        def apply(cls, transform_data: TransformData, redundant_temp_fields: bool):
            maker = cls()
            return maker(transform_data, redundant_temp_fields)

        def __call__(self, transform_data: TransformData, redundant_temp_fields: bool):
            self.data = transform_data
            self.redundant_temp_fields = redundant_temp_fields
            self.visit(self.data.definition_ir)

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

    class BlockMaker(gt_ir.IRNodeVisitor):
        """Creates the block tree consisting of DomainBlockInfo, IJBlockInfo,
        IntervalBlockInfo, and StatementInfo."""

        def __init__(self, computation_intervals: list):
            self.computation_intervals = computation_intervals

        @classmethod
        def apply(cls, transform_data: TransformData, computation_intervals: list):
            maker = cls(computation_intervals)
            return maker(transform_data)

        def __call__(self, transform_data: TransformData):
            self.data = transform_data
            self.current_block_info = None
            self.zero_extent = Extent.zeros(transform_data.ndims)
            self.visit(self.data.definition_ir)

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
            """Traverse statements, creating new IJBlockInfo where non-zero
            accesses to output fields are made."""
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
            """Creates a DomainBlockInfo for every interval and every computation."""
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

            # Merge offsets for same symbol
            for name, extent in refs:
                extent = extent or Extent.zeros()
                result.setdefault(name, Extent.zeros())
                result[name] |= extent

            return result

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._DEFAULT_OPTIONS

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        computation_intervals = cls.make_k_intervals(transform_data)
        cls.SymbolMaker.apply(transform_data, cls._DEFAULT_OPTIONS["redundant_temp_fields"])
        cls.BlockMaker.apply(transform_data, computation_intervals)


class NormalizeBlocksPass(TransformPass):
    """Create a DomainBlockInfo for each StatementInfo.

    Note
    ----
    The following `transform_data` attributes are changed:
        - `blocks`: DomainBlockInfo each contain only a single StatementInfo.
    """

    _DEFAULT_OPTIONS = {}

    class SplitBlocksVisitor:
        def __init__(self):
            self._split_blocks = []

        def visit(self, transform_data: TransformData) -> List[DomainBlockInfo]:
            for block in transform_data.blocks:
                context = {
                    "zero_extent": Extent.zeros(transform_data.ndims),
                    "id_generator": transform_data.id_generator,
                }
                self.visit_DomainBlockInfo(block, context)

            return self._split_blocks

        def visit_DomainBlockInfo(self, block: DomainBlockInfo, context: Dict[str, Any]) -> None:
            context["iteration_order"] = block.iteration_order
            for ij_block in block.ij_blocks:
                self.visit_IJBlockInfo(ij_block, context)

        def visit_IJBlockInfo(self, ij_block: IJBlockInfo, context: Dict[str, Any]) -> None:
            for interval_block in ij_block.interval_blocks:
                self.visit_IntervalBlockInfo(interval_block, context)

        def visit_IntervalBlockInfo(
            self, interval_block: IntervalBlockInfo, context: Dict[str, Any]
        ) -> None:
            context["interval"] = interval_block.interval
            for statement in interval_block.stmts:
                self.visit_StatemenInfo(statement, context)

        def visit_StatemenInfo(
            self, statement: StatementInfo, context: Dict[str, Any]
        ) -> DomainBlockInfo:
            new_interval_block = IntervalBlockInfo(
                context["id_generator"].new,
                context["interval"],
                [statement],
                statement.inputs,
                statement.outputs,
            )
            new_ij_block = IJBlockInfo(
                context["id_generator"].new,
                {context["interval"]},
                [new_interval_block],
                {**new_interval_block.inputs},
                set(new_interval_block.outputs),
                compute_extent=context["zero_extent"],
            )
            new_block = DomainBlockInfo(
                context["id_generator"].new,
                context["iteration_order"],
                set(new_ij_block.intervals),
                [new_ij_block],
                {**new_ij_block.inputs},
                set(new_ij_block.outputs),
            )
            self._split_blocks.append(new_block)

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._DEFAULT_OPTIONS

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        transform_data.blocks = cls.SplitBlocksVisitor().visit(transform_data)


class MultiStageMergingWrapper:
    """Wrapper for :class:`DomainBlockInfo` containing the logic required to merge or not merge."""

    def __init__(self, multi_stage: DomainBlockInfo, parent: TransformData):
        self._multi_stage = multi_stage
        self._parent = parent

    @classmethod
    def wrap_items(
        cls, items: Sequence[DomainBlockInfo], *, parent: TransformData
    ) -> List["MultiStageMergingWrapper"]:
        return [cls(block, parent) for block in items]

    def can_merge_with(self, candidate: "MultiStageMergingWrapper") -> bool:
        if self.parent != candidate.parent:
            return False
        if candidate.iteration_order != self.iteration_order:
            return False
        if candidate.has_disallowed_read_after_write_in(self):
            return False
        if candidate.has_disallowed_write_after_read_in(self):
            return False
        return True

    def merge_with(self, candidate: "MultiStageMergingWrapper") -> None:
        self._multi_stage.id = self._parent.id_generator.new
        self._multi_stage.ij_blocks.extend(candidate.ij_blocks)
        self._multi_stage.intervals |= candidate.intervals
        self._multi_stage.outputs |= candidate.outputs
        for name, extent in candidate.inputs.items():
            if name in self.inputs:
                self._multi_stage.inputs[name] |= extent
            else:
                self._multi_stage.inputs[name] = extent

    def has_disallowed_read_after_write_in(self, target: "MultiStageMergingWrapper") -> bool:
        if not self.k_offset_extends_domain:
            return False
        return any(extent[-1] != (0, 0) for extent in self.read_after_write_extents_in(target))

    def has_disallowed_write_after_read_in(self, target: "MultiStageMergingWrapper") -> bool:
        write_after_read_fields = self.write_after_read_fields_in(target)
        return write_after_read_fields and (
            self.has_reads_with_offset(restrict_to=write_after_read_fields)
            or target.has_reads_with_offset(restrict_to=write_after_read_fields)
            or self.has_extended_domain
            or target.has_extended_domain
        )

    def has_reads_with_offset(self, *, restrict_to: Optional[Set[str]]) -> bool:
        checked_axes = slice(None) if self.k_offset_extends_domain else slice(None, -1)
        fields = restrict_to.intersection(self.inputs) if restrict_to else set(self.inputs)
        return any(
            self.inputs[name][checked_axes] != Extent.zeros()[checked_axes] for name in fields
        )

    def read_after_write_fields_in(self, target: "MultiStageMergingWrapper") -> Set[str]:
        previous_writes = set(target.outputs)
        current_reads = set(self.inputs)
        return previous_writes.intersection(current_reads)

    def write_after_read_fields_in(self, target: "MultiStageMergingWrapper") -> Set[str]:
        previous_reads = set(target.inputs)
        current_writes = set(self.outputs)
        return previous_reads.intersection(current_writes)

    def read_after_write_extents_in(self, target: "MultiStageMergingWrapper") -> Set[Extent]:
        return {self.inputs[name] for name in self.read_after_write_fields_in(target)}

    @staticmethod
    def accumulate_extents(extents: Sequence[Extent]) -> Extent:
        full_extent = Extent.zeros()
        for extent in extents:
            full_extent |= extent
        return full_extent

    @property
    def extents(self) -> Iterator[Extent]:
        return (ij_block.compute_extent for ij_block in self.ij_blocks)

    @property
    def full_extent(self) -> Extent:
        return self.accumulate_extents(self.extents)

    @property
    def has_extended_domain(self) -> bool:
        return self.full_extent != Extent.zeros()

    @property
    def k_offset_extends_domain(self) -> bool:
        return (
            self.iteration_order == gt_ir.IterationOrder.PARALLEL
            and self._parent.has_sequential_axis
        )

    @property
    def iteration_order(self) -> gt_ir.IterationOrder:
        return self._multi_stage.iteration_order

    @property
    def inputs(self) -> Dict[str, Extent]:
        return self._multi_stage.inputs

    @property
    def intervals(self) -> List[IntervalInfo]:
        return self._multi_stage.intervals

    @property
    def ij_blocks(self) -> List[IJBlockInfo]:
        return self._multi_stage.ij_blocks

    @property
    def outputs(self) -> Dict[str, Extent]:
        return self._multi_stage.outputs

    @property
    def parent(self) -> TransformData:
        return self._parent

    @property
    def wrapped(self) -> DomainBlockInfo:
        return self._multi_stage


class StageMergingWrapper:
    """Wrapper for :class:`IJBlockInfo` containing the logic required to merge or not merge."""

    def __init__(self, stage: IJBlockInfo, parent: TransformData, parent_block: DomainBlockInfo):
        self._stage = stage
        self._parent = parent
        self._parent_block = parent_block

    @classmethod
    def wrap_items(
        cls, items: Sequence[IJBlockInfo], *, parent: TransformData, parent_block: DomainBlockInfo
    ) -> List["StageMergingWrapper"]:
        return [cls(ij_block, parent, parent_block) for ij_block in items]

    def can_merge_with(self, candidate: "StageMergingWrapper") -> bool:
        if not self.parent_block == candidate.parent_block:
            return False

        # Check that the two stages have the same compute extent
        if not (self.compute_extent == candidate.compute_extent):
            return False

        # Check that there is not overlap between stage intervals and that
        # merging stages will not imply a reordering of the execution order
        if self.has_incompatible_intervals_with(candidate):
            return False

        # Check that there are not data dependencies between stages
        if self.has_data_dependencies_with(candidate):
            return False
        return True

    def merge_with(self, candidate: IJBlockInfo):
        self._merge_interval_blocks_with(candidate)

        self._stage.intervals |= candidate.intervals
        self._stage.outputs |= candidate.outputs
        for name, extent in candidate.inputs.items():
            if name in self.inputs:
                self._stage.inputs[name] |= extent
            else:
                self._stage.inputs[name] = extent

    def _merge_interval_blocks_with(self, candidate: IJBlockInfo) -> None:
        i_to_ib_map = self.interval_to_iblock_mapping
        for candidate_iblock in candidate.interval_blocks:
            if candidate_iblock.interval in i_to_ib_map:
                self._merge_interval_block(
                    i_to_ib_map[candidate_iblock.interval], candidate_iblock
                )
            else:
                self._stage.interval_blocks.append(candidate_iblock)

    def _merge_interval_block(self, target_iblock, candidate_iblock: IntervalBlockInfo) -> None:
        target_iblock.id = self.parent.id_generator.new
        target_iblock.stmts.extend(candidate_iblock.stmts)
        self._merge_inputs(target_iblock.inputs, candidate_iblock.inputs)
        target_iblock.outputs |= candidate_iblock.outputs

    @staticmethod
    def _merge_inputs(
        target_inputs: Dict[str, Extent], candidate_inputs: Dict[str, Extent]
    ) -> None:
        for name, extent in candidate_inputs.items():
            if name in target_inputs:
                target_inputs[name] |= extent
            else:
                target_inputs[name] = extent

    def has_incompatible_intervals_with(self, candidate: "StageMergingWrapper") -> bool:
        for interval, candidate_interval in itertools.product(self.intervals, candidate.intervals):
            if self.intervals_overlap_or_imply_reorder(interval, candidate_interval):
                return True
        return False

    def has_data_dependencies_with(self, candidate: "StageMergingWrapper") -> bool:
        extents = (extent for name, extent in candidate.inputs.items() if name in self.outputs)
        for extent in extents:
            read_interval = (
                next(iter(self.intervals)).as_tuple(self.min_k_interval_sizes) + extent[-1]
            )
            for merged_interval_block in self.interval_blocks:
                merged_interval = merged_interval_block.interval
                if merged_interval.as_tuple(
                    self.min_k_interval_sizes
                ) != read_interval and merged_interval.overlaps(
                    read_interval, self.min_k_interval_sizes
                ):
                    return True
        return False

    def intervals_overlap_or_imply_reorder(
        self,
        interval_a: IntervalInfo,
        interval_b: IntervalInfo,
    ) -> bool:
        return interval_a != interval_b and interval_a.overlaps(
            interval_b, self.min_k_interval_sizes
        )

    @property
    def parent_block(self) -> TransformData:
        return self._parent_block

    @property
    def compute_extent(self) -> Extent:
        return self._stage.compute_extent

    @property
    def intervals(self) -> List[IntervalInfo]:
        return self._stage.intervals

    @property
    def interval_blocks(self) -> List[IntervalBlockInfo]:
        return self._stage.interval_blocks

    @property
    def interval_to_iblock_mapping(self) -> Dict[IntervalInfo, IntervalBlockInfo]:
        return {iblock.interval: iblock for iblock in self.interval_blocks}

    @property
    def inputs(self) -> Dict[str, Extent]:
        return self._stage.inputs

    @property
    def outputs(self) -> Dict[str, Extent]:
        return self._stage.outputs

    @property
    def min_k_interval_sizes(self) -> int:
        return self._parent.min_k_interval_sizes

    @property
    def parent(self) -> TransformData:
        return self._parent

    @property
    def wrapped(self) -> IJBlockInfo:
        return self._stage


def greedy_merging(items: Sequence[MergeableType]) -> List[MergeableType]:
    if len(items) < 2:
        return items
    merged_items = [items[0]]
    for candidate in items[1:]:
        target = merged_items[-1]
        if target.can_merge_with(candidate):
            target.merge_with(candidate)
        else:
            merged_items.append(candidate)
    return merged_items


def greedy_merging_with_wrapper(
    items: Sequence[WrappedType], wrapper_cls: Type[MergeableType], **kwargs: Any
) -> List[WrappedType]:
    return [w.wrapped for w in greedy_merging(wrapper_cls.wrap_items(items, **kwargs))]


class MergeBlocksPass(TransformPass):
    """Merges `transform_data.blocks` using a greedy algorithm.

    The first step merges IJBlockInfos as long as compatibility conditions are met, then proceeds to try and merge
    IJBlockInfos. The secondary merging step attempts to create as few IntervalBlockInfos as necessary, by re-using
    existing blocks with the same interval. Note that this could be re-implemented as a third merging step for every
    IJBlockInfo instead.

    Note
    ----
    The following `transform_data` attributes are changed:
        - `blocks`: Merged as far as possible without reordering.
    """

    _DEFAULT_OPTIONS = {}

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._DEFAULT_OPTIONS

    @staticmethod
    def apply(transform_data: TransformData) -> None:
        merged_blocks = greedy_merging_with_wrapper(
            transform_data.blocks, MultiStageMergingWrapper, parent=transform_data
        )
        for block in merged_blocks:
            block.ij_blocks = greedy_merging_with_wrapper(
                block.ij_blocks, StageMergingWrapper, parent=transform_data, parent_block=block
            )
        transform_data.blocks = merged_blocks


class ComputeExtentsPass(TransformPass):
    """Loop over blocks backwards and accumulate extents.

    Note
    ----
    Writes to `transform_data.blocks` and fills each IJBlockInfo.compute_extent,
    and creates `transform_data.implementation_ir.fields_extent`.
    """

    _DEFAULT_OPTIONS = {}

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._DEFAULT_OPTIONS

    @staticmethod
    def apply(transform_data: TransformData) -> None:
        seq_axis = transform_data.definition_ir.domain.index(
            transform_data.definition_ir.domain.sequential_axis
        )
        access_extents = {}
        for name in transform_data.symbols:
            access_extents[name] = Extent.zeros()

        blocks = transform_data.blocks
        for dom_block in reversed(blocks):
            for ij_block in reversed(dom_block.ij_blocks):
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


class DataTypePass(TransformPass):
    """Fills in the concrete data_type for all set to `DataType.AUTO`"""

    class CollectDataTypes(gt_ir.IRNodeVisitor):
        def __call__(self, node) -> None:
            assert isinstance(node, gt_ir.StencilImplementation)
            self.vars = node.parameters
            self.fields = node.fields
            self.visit(node)

        @classmethod
        def apply(cls, node) -> None:
            collector = cls()
            return collector(node)

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

            dtypes_set = set(
                arg.data_type for arg in node.args if arg.data_type != gt_ir.DataType.DEFAULT
            )

            if len(dtypes_set) == 0:
                data_type = gt_ir.DataType.DEFAULT
            elif len(dtypes_set) == 1:
                data_type = dtypes_set.pop()
            else:
                # get the "largest" data type in the set
                data_type = gt_ir.DataType.merge(*dtypes_set)
                # cast all other args to this type
                for index, arg in enumerate(node.args):
                    if arg.data_type != data_type:
                        node.args[index] = gt_ir.Cast(dtype=data_type, expr=arg, loc=node.loc)

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

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        cls.CollectDataTypes.apply(transform_data.implementation_ir)


class ComputeUsedSymbolsPass(TransformPass):
    """Fills the SymbolInfo `in_use` attribute in `transform_data.symbols`.

    It does this because an entry was originally created for each Decl in the
    SymbolMaker part of InitInfoPass, and some of these may not be referenced.
    """

    class ComputeUsedVisitor(gt_ir.IRNodeVisitor):
        def __init__(self, transform_data: TransformData):
            self.data = transform_data

        def visit_VarRef(self, node: gt_ir.VarRef, **kwargs):
            self.data.symbols[node.name].in_use = True

        def visit_FieldRef(self, node: gt_ir.FieldRef, **kwargs):
            self.data.symbols[node.name].in_use = True

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        visitor = cls.ComputeUsedVisitor(transform_data)
        visitor.visit(transform_data.definition_ir)


class BuildIIRPass(TransformPass):
    """Transcribe `transform_data.blocks` to `transform_data.implementation_ir`.

    Note
    ----
    DomainBlockInfo -> MultiStage
    IJBlockInfo -> Stage
    IntervalBlockInfo -> ApplyBlock
    """

    _DEFAULT_OPTIONS = {}

    def __init__(self):
        self.data: TransformData = None
        self.iir: gt_ir.StencilImplementation = None

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._DEFAULT_OPTIONS

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        return cls()(transform_data)

    def __call__(self, transform_data: TransformData) -> None:
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

    def _make_stage(self, ij_block) -> gt_ir.Stage:
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

    def _make_apply_block(self, interval_block) -> gt_ir.ApplyBlock:
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

    def _make_accessor(self, name, extent, read_write: bool) -> gt_ir.Accessor:
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

    def _make_axis_interval(self, interval: IntervalInfo) -> gt_ir.AxisInterval:
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
    """Demote symbols only used within a single stage to scalars.

    This may occur because these can be local variables in the scope, and
    therefore do not need to be fields.
    """

    class CollectDemotableSymbols(gt_ir.IRNodeVisitor):
        def __init__(self):
            pass

        @classmethod
        def apply(cls, node: gt_ir.StencilImplementation) -> Set[str]:
            collector = cls()
            return collector(node)

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
        @classmethod
        def apply(cls, node: gt_ir.StencilImplementation, demotables: Set[str]) -> None:
            instance = cls(demotables)
            return instance(node)

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

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        demotables = cls.CollectDemotableSymbols.apply(transform_data.implementation_ir)
        cls.DemoteSymbols.apply(transform_data.implementation_ir, demotables)


class HousekeepingPass(TransformPass):
    class WarnIfNoEffect(gt_ir.IRNodeVisitor):
        """Warn if StencilImplementation has no effect."""

        def __init__(self):
            pass

        @classmethod
        def apply(cls, stencil_name: str, node: gt_ir.StencilImplementation) -> None:
            instance = cls()
            return instance(stencil_name, node)

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
        """Removes empty multi-stages, stage groups, and stages."""

        def __init__(self):
            pass

        @classmethod
        def apply(cls, node: gt_ir.StencilImplementation) -> None:
            instance = cls()
            return instance(node)

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

    @classmethod
    def apply(cls, transform_data: TransformData) -> None:
        cls.PruneEmptyNodes.apply(transform_data.implementation_ir)
        cls.WarnIfNoEffect.apply(
            transform_data.definition_ir.name, transform_data.implementation_ir
        )
