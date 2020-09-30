from typing import Tuple

from gt4py.ir.nodes import Domain, IterationOrder

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition


def test_write_after_read_ij_extended(
    normalize_blocks_pass: AnalysisPass,
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
    transform_data = normalize_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 3


def test_write_after_read_ij_offset(
    normalize_blocks_pass: AnalysisPass,
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
    transform_data = normalize_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 2
