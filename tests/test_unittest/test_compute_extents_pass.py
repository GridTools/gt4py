from gt4py.analysis.infos import IntervalInfo
from gt4py.analysis.passes import ComputeExtentsPass
from gt4py.ir.nodes import AxisBound, AxisInterval, Domain, IterationOrder, LevelMarker

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition


def test_intervalinfo_overlap():
    overlap = ComputeExtentsPass.overlap_with_extent(
        IntervalInfo(start=(0, -1), end=(0, 2)), (0, 2)
    )
    assert overlap == (0, IntervalInfo.MAX_INT)

    overlap = ComputeExtentsPass.overlap_with_extent(
        IntervalInfo(start=(0, -1), end=(1, 0)), (0, 2)
    )
    assert overlap == (0, 2)

    overlap = ComputeExtentsPass.overlap_with_extent(
        IntervalInfo(start=(0, -3), end=(0, -1)), (0, 0)
    )
    assert overlap is None

    overlap = ComputeExtentsPass.overlap_with_extent(
        IntervalInfo(start=(1, -3), end=(1, 0)), (0, 1)
    )
    assert overlap == (-IntervalInfo.MAX_INT, 1)

    overlap = ComputeExtentsPass.overlap_with_extent(
        IntervalInfo(start=(1, -1), end=(1, 0)), (0, 0)
    )
    assert overlap == (-IntervalInfo.MAX_INT, 0)

    overlap = ComputeExtentsPass.overlap_with_extent(
        IntervalInfo(start=(0, 2), end=(0, 3)), (0, 0)
    )
    assert overlap == (-2, IntervalInfo.MAX_INT)


def test_simple(
    compute_extents_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "in", "tmp"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "in", (-1, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("out", "tmp", (-2, 0, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = compute_extents_pass(transform_data)
    impl_ir = transform_data.implementation_ir

    block_extents = [block.ij_blocks[0].compute_extent[0] for block in transform_data.blocks]
    correct_extents = [(-2, 0), (0, 0)]
    assert all(block == correct for block, correct in zip(block_extents, correct_extents))

    correct_field_extents = {"in": (-3, 0), "tmp": (-2, 0), "out": (0, 0)}
    assert all(
        correct_field_extents[field] == extent[0]
        for field, extent in impl_ir.fields_extents.items()
    )


def test_parallel_interval_dependency(
    compute_extents_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "inout", "in", "in2"])
        .add_blocks(
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("inout", "in", (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("tmp", "in2", (0, 0, 0)),
            ),
            TComputationBlock(order=IterationOrder.PARALLEL)
            .add_statements(
                TAssign("inout", "tmp", (-4, 0, 0)),
            )
            .with_parallel_interval(
                AxisInterval(
                    start=AxisBound(level=LevelMarker.START, offset=1),
                    end=AxisBound(level=LevelMarker.START, offset=2),
                ),
                None,
            ),
            TComputationBlock(order=IterationOrder.PARALLEL).add_statements(
                TAssign("out", "inout", (-2, 0, 0)),
            ),
        )
        .build_transform()
    )
    transform_data = compute_extents_pass(transform_data)
    impl_ir = transform_data.implementation_ir

    correct_extents = [(-2, 0), (-3, 0), (-2, 0), (0, 0)]
    block_extents = [block.ij_blocks[0].compute_extent[0] for block in transform_data.blocks]
    assert all(block == correct for block, correct in zip(block_extents, correct_extents))

    correct_field_extents = {
        "out": (0, 0),
        "inout": (-2, 0),
        "tmp": (-3, 0),
        "in": (-2, 0),
        "in2": (-3, 0),
    }
    assert all(
        correct_field_extents[field] == extent[0]
        for field, extent in impl_ir.fields_extents.items()
    )


def test_write_consume_parallel_interval(
    compute_extents_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "in", "tmp"])
        .add_blocks(
            # fill in region
            TComputationBlock(order=IterationOrder.PARALLEL)
            .add_statements(
                TAssign("tmp", "in", (0, 0, 0)),
            )
            .with_parallel_interval(
                AxisInterval(
                    start=AxisBound(level=LevelMarker.START, offset=-1),
                    end=AxisBound(level=LevelMarker.START, offset=0),
                ),
                None,
            ),
            # consume in region
            TComputationBlock(order=IterationOrder.PARALLEL)
            .add_statements(
                TAssign("out", "tmp", (-2, 0, 0)),
            )
            .with_parallel_interval(
                AxisInterval(
                    start=AxisBound(level=LevelMarker.START, offset=1),
                    end=AxisBound(level=LevelMarker.START, offset=2),
                ),
                None,
            ),
        )
        .build_transform()
    )

    transform_data = compute_extents_pass(transform_data)
    impl_ir = transform_data.implementation_ir

    correct_extents = [(-1, 0), (0, 0)]
    block_extents = [block.ij_blocks[0].compute_extent[0] for block in transform_data.blocks]
    assert all(block == correct for block, correct in zip(block_extents, correct_extents))

    correct_field_extents = {
        "out": (0, 0),
        "in": (-1, 0),
        "tmp": (-1, 0),
    }
    assert all(
        correct_field_extents[field] == extent[0]
        for field, extent in impl_ir.fields_extents.items()
    )


def test_remove_interval(
    compute_extents_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            # fill in region
            TComputationBlock(order=IterationOrder.PARALLEL)
            .add_statements(
                TAssign("out", "in", (0, 0, 0)),
            )
            .with_parallel_interval(
                AxisInterval(
                    start=AxisBound(level=LevelMarker.START, offset=-1),
                    end=AxisBound(level=LevelMarker.START, offset=0),
                ),
                None,
            ),
        )
        .build_transform()
    )

    transform_data = compute_extents_pass(transform_data)
    assert len(transform_data.blocks[0].ij_blocks) == 0


def test_end_interval_extent(
    compute_extents_pass: AnalysisPass,
    ijk_domain: Domain,
) -> None:
    transform_data = (
        TDefinition(name="ij_extended", domain=ijk_domain, fields=["out", "in"])
        .add_blocks(
            # fill in region
            TComputationBlock(order=IterationOrder.PARALLEL)
            .add_statements(
                TAssign("out", "in", (1, 0, 0)),
            )
            .with_parallel_interval(
                AxisInterval(
                    start=AxisBound(level=LevelMarker.END, offset=-1),
                    end=AxisBound(level=LevelMarker.END, offset=0),
                ),
                None,
            ),
        )
        .build_transform()
    )

    transform_data = compute_extents_pass(transform_data)
    assert transform_data.implementation_ir.fields_extents["in"][0] == (0, 1)
    assert transform_data.blocks[0].ij_blocks[0].inputs["in"][0] == (0, 1)
    assert transform_data.blocks[0].inputs["in"][0] == (0, 1)
