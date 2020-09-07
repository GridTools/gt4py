from typing import Tuple

from gt4py.ir.nodes import Domain, IterationOrder

from ..analysis_setup import PassType
from ..definition_setup import make_transform_data


def test_merge_write_after_read_ij_extended(
    merge_blocks_pass: PassType,
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
    transform_data = merge_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 2
    assert len(transform_data.blocks[0].ij_blocks) == 2
    assert len(transform_data.blocks[1].ij_blocks) == 1


def test_merge_write_after_read_ij_offset(
    merge_blocks_pass: PassType,
    iteration_order: IterationOrder,
    ij_offset: Tuple[int, int, int],
    ijk_domain: Domain,
) -> None:
    transform_data = make_transform_data(
        name="ij_offset",
        domain=ijk_domain,
        fields=["out", "inout", "in"],
        body=[("out", "inout", ij_offset), ("inout", "in", (0, 0, 0))],
        iteration_order=iteration_order,
    )
    transform_data = merge_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 2
    assert len(transform_data.blocks[0].ij_blocks) == 1
    assert len(transform_data.blocks[1].ij_blocks) == 1


def test_merge_read_after_read_ij_offset(
    merge_blocks_pass: PassType,
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
    transform_data = merge_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 1


def test_merge_write_after_read_k(
    merge_blocks_pass: PassType, non_parallel_iteration_order: IterationOrder, ijk_domain: Domain
) -> None:
    transform_data = make_transform_data(
        name="k_offset_nonparallel",
        domain=ijk_domain,
        fields=["out", "inout", "in"],
        body=[("out", "inout", (0, 0, -1)), ("inout", "in", (0, 0, 0))],
        iteration_order=non_parallel_iteration_order,
    )
    transform_data = merge_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 1


def test_merge_write_after_read_k_parallel(merge_blocks_pass, ijk_domain):
    transform_data = make_transform_data(
        name="k_offset_parallel",
        domain=ijk_domain,
        fields=["out", "inout", "in"],
        body=[("out", "inout", (0, 0, -1)), ("inout", "in", (0, 0, 0))],
        iteration_order=IterationOrder.PARALLEL,
    )
    transform_data = merge_blocks_pass(transform_data)
    assert len(transform_data.blocks) == 2
