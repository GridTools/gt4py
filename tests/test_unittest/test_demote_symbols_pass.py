# -*- coding: utf-8 -*-
from gt4py.ir.nodes import Domain, IterationOrder

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition


def test_demote_locals_no_extents(
    demote_locals_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="no_offsets", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("tmp", "in", (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, 0)),
            )
        )
        .build_transform()
    )
    transform_data = demote_locals_pass(transform_data)
    assert not transform_data.implementation_ir.temporary_fields


def test_demote_locals_ij_offsets(
    demote_locals_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_offsets", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=iteration_order).add_statements(
                TAssign("tmp", "in", (-1, 1, 0)),
                TAssign("out", "tmp", (-1, 1, 0)),
            )
        )
        .build_transform()
    )
    transform_data = demote_locals_pass(transform_data)
    assert transform_data.implementation_ir.temporary_fields == ["tmp"]


def test_demote_locals_two_intervals(
    demote_locals_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="two_intervals", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-1).add_statements(
                TAssign("tmp", "in", (0, 0, 1)),
                TAssign("out", "tmp", (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-2).add_statements(
                TAssign("tmp", "in", (0, 0, -1)),
                TAssign("out", "tmp", (0, 0, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = demote_locals_pass(transform_data)
    assert not transform_data.implementation_ir.temporary_fields


def test_demote_locals_two_intervals_ij_offsets(
    demote_locals_pass: AnalysisPass,
    iteration_order: IterationOrder,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="two_intervals_ij_offsets", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-1).add_statements(
                TAssign("tmp", "in", (-1, 1, 1)),
                TAssign("out", "tmp", (-1, 1, 0)),
            ),
            TComputationBlock(order=IterationOrder.FORWARD, start=1, end=-2).add_statements(
                TAssign("tmp", "in", (-1, 1, -1)),
                TAssign("out", "tmp", (-1, 1, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = demote_locals_pass(transform_data)
    assert transform_data.implementation_ir.temporary_fields == ["tmp"]


def test_do_not_demote_self_assign(
    demote_locals_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="self_assignment", domain=ijk_domain, fields=["out", "input"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "input", (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.FORWARD).add_statements(
                TAssign("out", "input", (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "tmp", (0, 0, 0)),
                TAssign("out", "tmp", (0, 0, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = demote_locals_pass(transform_data)
    assert transform_data.implementation_ir.temporary_fields == ["tmp"]
