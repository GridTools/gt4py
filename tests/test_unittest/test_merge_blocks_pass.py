from typing import Tuple

import pytest

from gt4py.ir.nodes import Axis, Domain, IterationOrder

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
    # not allowed to be merged: race condition independent of iteration order
    # write after read with extended compute domain
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
    # not allowed to be merged: race condition independent of iteration order
    # write after read: no input with offset allowed
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
    #  allowed to be merged: no write to "in"
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
    # allowed to be merged: order is sequential -> no race condition on k-offsets
    assert len(transform_data.blocks) == 1


def test_merge_write_after_read_k_parallel(
    merge_blocks_pass: PassType, ijk_domain: Domain
) -> None:
    transform_data = make_transform_data(
        name="k_offset_parallel",
        domain=ijk_domain,
        fields=["out", "inout", "in"],
        body=[("out", "inout", (0, 0, -1)), ("inout", "in", (0, 0, 0))],
        iteration_order=IterationOrder.PARALLEL,
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged: order is PARALLEL -> race condition even on k-offsets
    assert len(transform_data.blocks) == 2


def test_merge_write_after_read_k_extended_sequential(
    merge_blocks_pass: PassType, non_parallel_iteration_order: IterationOrder, ijk_domain: Domain
) -> None:
    transform_data = make_transform_data(
        name="k_extended",
        domain=ijk_domain,
        fields=["out", "in", "inout"],
        body=[("tmp", "inout", (0, 0, 0)), ("out", "tmp", (0, 0, 1)), ("inout", "in", (0, 0, 0))],
        iteration_order=non_parallel_iteration_order,
    )
    transform_data = merge_blocks_pass(transform_data)
    # should be merged, since k-offset in sequential order does not extend compute domain
    assert len(transform_data.blocks) == 1


def test_merge_read_after_write_k_parallel_seq(
    merge_blocks_pass: PassType, ijk_domain: Domain
) -> None:
    transform_data = make_transform_data(
        name="read_after_write_forbidden",
        domain=ijk_domain,
        fields=["out", "in"],
        body=[("tmp", "in", (0, 0, 0)), ("out", "tmp", (0, 0, -1))],
        iteration_order=IterationOrder.PARALLEL,
    )
    transform_data = merge_blocks_pass(transform_data)
    # not allowed to be merged, because PARALLEL and k-axis is sequential
    assert len(transform_data.blocks) == 2


@pytest.mark.skip(reason="ComputeExtentsPass fails if no sequential axis")
def test_merge_read_after_write_k_parallel_noseq(merge_blocks_pass: PassType) -> None:
    transform_data = make_transform_data(
        name="read_after_write_forbidden",
        # type ignore is due to attribclass
        domain=Domain(parallel_axes=[Axis(name=idx) for idx in ["I", "J", "K"]]),  # type: ignore
        fields=["out", "in"],
        body=[("tmp", "in", (0, 0, 0)), ("out", "tmp", (0, 0, -1))],
        iteration_order=IterationOrder.PARALLEL,
    )
    transform_data = merge_blocks_pass(transform_data)
    # allowed to be merged, because k-axis is not sequential
    assert len(transform_data.blocks) == 1


def test_merge_read_after_write_k_sequential(
    merge_blocks_pass: PassType, ijk_domain: Domain, non_parallel_iteration_order: IterationOrder
) -> None:
    transform_data = make_transform_data(
        name="read_after_write_forbidden",
        domain=ijk_domain,
        fields=["out", "in"],
        body=[("tmp", "in", (0, 0, 0)), ("out", "tmp", (0, 0, -1))],
        iteration_order=non_parallel_iteration_order,
    )
    transform_data = merge_blocks_pass(transform_data)
    # allowed to be merged, because order is not PARALLEL
    assert len(transform_data.blocks) == 1
