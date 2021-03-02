# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

from collections import namedtuple
from typing import List, Tuple

import pytest

from gt4py.analysis import (
    DomainBlockInfo,
    IJBlockInfo,
    IntervalBlockInfo,
    StatementInfo,
    TransformData,
)
from gt4py.ir.nodes import Axis, Domain, IterationOrder

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition


def test_merge_write_after_read_ij_extended(
    merge_blocks_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "in", "inout"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("tmp", "inout", (0, 0, 0)),
                TAssign("out", "tmp", ij_offset),
                TAssign("inout", "in", (0, 0, 0)),
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged: race condition independent of iteration order
    # write after read with extended compute domain
    assert len(transform_data.blocks) == 2
    assert len(transform_data.blocks[0].ij_blocks) == 2
    assert len(transform_data.blocks[1].ij_blocks) == 1


def test_merge_write_after_read_ij_offset(
    merge_blocks_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_offset", domain=ijk_domain, fields=["out", "inout", "in"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("out", "inout", ij_offset), TAssign("inout", "in", (0, 0, 0))
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged: race condition independent of iteration order
    # write after read: no input with offset allowed
    assert len(transform_data.blocks) == 2
    assert len(transform_data.blocks[0].ij_blocks) == 1
    assert len(transform_data.blocks[1].ij_blocks) == 1


def test_merge_read_after_read_ij_offset(
    merge_blocks_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_offset_readonly", domain=ijk_domain, fields=["out", "in", "inout"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("out", "in", ij_offset), TAssign("inout", "in", (0, 0, 0))
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    #  allowed to be merged: no write to "in"
    assert len(transform_data.blocks) == 1


def test_merge_write_after_read_k(
    merge_blocks_pass: AnalysisPass,
    non_parallel_iteration_order: IterationOrder,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="k_offset_nonparallel", domain=ijk_domain, fields=["out", "inout", "in"])
        .add_blocks(
            TComputationBlock(order=non_parallel_iteration_order).add_statements(
                TAssign("out", "inout", (0, 0, -1)), TAssign("inout", "in", (0, 0, 0))
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # allowed to be merged: order is sequential -> no race condition on k-offsets
    assert len(transform_data.blocks) == 1


def test_merge_write_after_read_k_parallel(
    merge_blocks_pass: AnalysisPass, ijk_domain: Domain
) -> None:
    transform_data = (
        TDefinition(name="k_offset_parallel", domain=ijk_domain, fields=["out", "inout", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("out", "inout", (0, 0, -1)), TAssign("inout", "in", (0, 0, 0))
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged: order is PARALLEL -> race condition even on k-offsets
    assert len(transform_data.blocks) == 2


def test_no_merge_read_with_offset_after_write(
    merge_blocks_pass: AnalysisPass, ijk_domain: Domain
) -> None:
    transform_data = (
        TDefinition(name="no_merge_k_offset", domain=ijk_domain, fields=["out", "in", "tmp"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-1).add_statements(
                TAssign("tmp", "in", (0, 0, 1))
            ),
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-2).add_statements(
                TAssign("out", "tmp", (0, 0, -1))
            ),
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged into the same stage: first block should have 2 IJ blocks
    assert len(transform_data.blocks[0].ij_blocks) == 2
    assert "in" in transform_data.blocks[0].ij_blocks[0].inputs
    assert "tmp" in transform_data.blocks[0].ij_blocks[1].inputs


def test_merge_write_after_read_k_extended_sequential(
    merge_blocks_pass: AnalysisPass,
    non_parallel_iteration_order: IterationOrder,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="k_extended", domain=ijk_domain, fields=["out", "in", "inout"])
        .add_blocks(
            TComputationBlock(order=non_parallel_iteration_order).add_statements(
                TAssign("tmp", "inout", (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, 1)),
                TAssign("inout", "in", (0, 0, 0)),
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # should be merged, since k-offset in sequential order does not extend compute domain
    assert len(transform_data.blocks) == 1


def test_merge_read_after_write_k_parallel_seq(
    merge_blocks_pass: AnalysisPass, ijk_domain: Domain
) -> None:
    transform_data = (
        TDefinition(
            name="read_after_write_forbidden_parallel", domain=ijk_domain, fields=["out", "in"]
        )
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "in", (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, -1)),
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged, because PARALLEL and k-axis is sequential
    assert len(transform_data.blocks) == 2


@pytest.mark.skip(reason="ComputeExtentsPass fails if no sequential axis")
def test_merge_read_after_write_k_parallel_noseq(merge_blocks_pass: AnalysisPass) -> None:
    transform_data = (
        TDefinition(
            name="read_after_write_forbidden_noseq",
            # type ignores are due to attribclass
            domain=Domain(  # type: ignore
                parallel_axes=[Axis(name=idx) for idx in ["I", "J", "K"]]  # type: ignore
            ),
            fields=["out", "in"],
        )
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "in", (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, -1)),
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # allowed to be merged, because k-axis is not sequential
    assert len(transform_data.blocks) == 1


def test_merge_read_after_write_k_sequential(
    merge_blocks_pass: AnalysisPass,
    ijk_domain: Domain,
    non_parallel_iteration_order: IterationOrder,
) -> None:
    transform_data = (
        TDefinition(name="read_after_write_forbidden_seq", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=non_parallel_iteration_order).add_statements(
                TAssign("tmp", "in", (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, -1)),
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # allowed to be merged, because order is not PARALLEL
    assert len(transform_data.blocks) == 1


def test_no_merge_with_overlapping_intervals(
    merge_blocks_pass: AnalysisPass, ijk_domain: Domain
) -> None:
    transform_data = (
        TDefinition(
            name="overlapping_intervals_forbidden",
            domain=ijk_domain,
            fields=["out1", "out2", "in1", "in2"],
        )
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL, start=0, end=0).add_statements(
                TAssign("out1", "in1", (0, 0, 0))
            ),
            TComputationBlock(order=IterationOrder.PARALLEL, start=1, end=-1).add_statements(
                TAssign("out2", "in2", (0, 0, 0))
            ),
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged into the same stage: overlapping intervals
    assert len(transform_data.blocks) == 1
    assert len(transform_data.blocks[0].ij_blocks) == 2
    block = transform_data.blocks[0].ij_blocks[0]
    assert "in1" in block.inputs and "out1" in block.outputs
    block = transform_data.blocks[0].ij_blocks[1]
    assert "in2" in block.inputs and "out2" in block.outputs


SPTYPE = namedtuple("SPTYPE", ("multi_stage", "stage", "interval_block", "statements"))


class _StatementPositionVisitor:
    """Collect position info for each statement in a TransformData instance."""

    def __init__(self):
        self.statements = {}
        self.cursor = [-1, -1, -1, -1]

    def visit(self, transform_data: TransformData) -> List[SPTYPE]:
        """
        Collect statement coordinates.

        Returns
        -------
            counting is from zero
            * multi_stage: in which multi stage is the statement?
            * stage: in which stage is the statement?
            * interval_block: in which interval block is the statement?
            * statements: position relative to other statements in the interval block?
        """
        for block in transform_data.blocks:
            self.cursor[0] += 1
            self.visit_DomainBlockInfo(block)
        return self.statements

    def visit_DomainBlockInfo(self, block: DomainBlockInfo) -> None:
        for ij_block in block.ij_blocks:
            self.cursor[1] += 1
            self.visit_IJBlockInfo(ij_block)
        self.cursor[1] = -1

    def visit_IJBlockInfo(self, ij_block: IJBlockInfo) -> None:
        for interval_block in ij_block.interval_blocks:
            self.cursor[2] += 1
            self.visit_IntervalBlockInfo(interval_block)
        self.cursor[2] = -1

    def visit_IntervalBlockInfo(self, interval_block: IntervalBlockInfo) -> None:
        for statement in interval_block.stmts:
            self.cursor[3] += 1
            self.visit_StatemenInfo(statement)
        self.cursor[3] = -1

    def visit_StatemenInfo(self, statement: StatementInfo) -> None:
        self.statements[statement.stmt.loc.line] = SPTYPE(*self.cursor)


def test_split_reorderable(merge_blocks_pass: AnalysisPass, ijk_domain: Domain) -> None:
    """
    Statements separated by a write after read occurrence are split in separate multi stages.

    Examples
    --------
    .. code-block: python

        with computation(FORWARD):
            with interval(...):
                tmp = a[0, 0, -1]  # stmt (0)
                ##
                tmp2 = b[1, 0, 0]  # stmt (1)
                b = in             # stmt (2)
                ##
                a = tmp            # stmt (3)

        # last statement can be merged with first statement
        # only if reordered first
    """
    statements = [
        ("tmp", "a", (0, 0, -1)),
        ("tmp2", "b", (1, 0, 0)),
        ("b", "in", (0, 0, 0)),
        ("a", "tmp", (0, 0, 0)),
    ]
    stmt_to_line = [1, 2, 3, 4]
    line_to_statement = [0, 1, 2, 3]
    transform_data = (
        TDefinition(name="reorderable", domain=ijk_domain, fields=["a", "b", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD).add_statements(
                *(TAssign(*statements[i]) for i in line_to_statement)
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # second and third statements are split
    # first is merged with second
    # third is merged with fourth
    statement_pos = _StatementPositionVisitor().visit(transform_data)
    statement_pos = [statement_pos[i] for i in stmt_to_line]  # convert from lineno to stmt #
    assert len(transform_data.blocks) == 2
    # first multi stage contains stmts 0 and 1 in order
    assert statement_pos[0].multi_stage == 0
    assert statement_pos[1].multi_stage == 0
    assert statement_pos[0].statements == 0
    assert statement_pos[1].statements == 1
    # second multi stage contains stmts 2 and 3 in order
    assert statement_pos[2].multi_stage == 1
    assert statement_pos[3].multi_stage == 1
    assert statement_pos[2].statements == 0
    assert statement_pos[3].statements == 1


def test_split_preordered(merge_blocks_pass: AnalysisPass, ijk_domain: Domain) -> None:
    """
    Statements preordered to be on the same side of the write after read occurence can be merged.

    Examples
    --------
    .. code-block: python

        with computation(FORWARD):
            with interval(...):
                tmp = a[0, 0, -1]  # stmt (0)
                a = tmp            # stmt (3)
                ##
                tmp2 = b[1, 0, 0]  # stmt (1)
                b = in             # stmt (2)

        # In contrast with the "split_reorderable" case,
        # statements 1 and 2 can be merged
    """
    statements = [
        ("tmp", "a", (0, 0, -1)),
        ("tmp2", "b", (1, 0, 0)),
        ("b", "in", (0, 0, 0)),
        ("a", "tmp", (0, 0, 0)),
    ]
    stmt_to_line = [1, 3, 4, 2]
    line_to_statement = [0, 3, 1, 2]
    transform_data = (
        TDefinition(name="preordered", domain=ijk_domain, fields=["a", "b", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD).add_statements(
                *(TAssign(*statements[i]) for i in line_to_statement)
            )
        )
        .build_transform()
    )
    transform_data = merge_blocks_pass(transform_data)
    # third and fourth statements are split
    # first is merged with second and third
    # fourth stands alone
    statement_pos = _StatementPositionVisitor().visit(transform_data)
    statement_pos = [statement_pos[i] for i in stmt_to_line]  # convert from lineno to stmt #
    assert len(transform_data.blocks) == 2
    # first multi stage contains stmt 0, 3 and 1 in order
    assert statement_pos[0].multi_stage == 0
    assert statement_pos[3].multi_stage == 0
    assert statement_pos[1].multi_stage == 0
    assert statement_pos[0].statements == 0
    assert statement_pos[3].statements == 1
    assert statement_pos[1].statements == 2
    # second multi stage contain]s statement 2
    assert statement_pos[2].multi_stage == 1
    assert statement_pos[2].statements == 0
