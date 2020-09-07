from typing import Tuple

from gt4py.ir.nodes import Domain, IterationOrder

from ..analysis_setup import PassType
from ..definition_setup import make_transform_data


def test_write_after_read_ij_extended(
    normalize_blocks_pass: PassType,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = make_transform_data(
        name="ij_extended",
        domain=ijk_domain,
        fields=["out", "in", "inout"],
        body=[("tmp", "inout", (0, 0, 0)), ("out", "tmp", ij_offset), ("inout", "in", (0, 0, 0))],
        iteration_order=iteration_order,
    )
    transform_data = normalize_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 3


def test_write_after_read_ij_offset(
    normalize_blocks_pass: PassType,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = make_transform_data(
        name="ij_offset_readonly",
        domain=ijk_domain,
        fields=["out", "in", "inout"],
        body=[("out", "in", ij_offset), ("inout", "in", (0, 0, 0))],
        iteration_order=iteration_order,
    )
    assert len(transform_data.blocks) == 2
