from gt4py.ir.nodes import AxisBound, AxisInterval, Domain, IterationOrder, LevelMarker

from ..analysis_setup import AnalysisPass
from ..definition_setup import TAssign, TComputationBlock, TDefinition


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
        "in": (0, 0),
        "tmp": (-1, 0),
    }
    assert all(
        correct_field_extents[field] == extent[0]
        for field, extent in impl_ir.fields_extents.items()
    )
