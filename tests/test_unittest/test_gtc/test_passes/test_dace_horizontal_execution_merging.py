from gtc.passes.oir_dace_optimizations.api import optimize_horizontal_executions
from gtc.passes.oir_dace_optimizations.horizontal_execution_merging import GraphMerging

from ..oir_utils import (
    AssignStmtFactory,
    HorizontalExecutionFactory,
    StencilFactory,
    VerticalLoopSectionFactory,
)


def test_zero_extent_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(
                body=[assignment_0 := AssignStmtFactory(left__name="foo", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[assignment_1 := AssignStmtFactory(left__name="baz", right__name="bar")]
            ),
            HorizontalExecutionFactory(
                body=[assignment_2 := AssignStmtFactory(left__name="foo", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[assignment_3 := AssignStmtFactory(left__name="foo", right__name="baz")],
            ),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    assert len(transformed.horizontal_executions) == 1
    transformed_order = transformed.horizontal_executions[0].body
    assert transformed_order.index(assignment_0) < transformed_order.index(assignment_2)
    assert transformed_order.index(assignment_1) < transformed_order.index(assignment_3)
    assert transformed_order.index(assignment_2) < transformed_order.index(assignment_3)


def test_mixed_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[assignment_0 := AssignStmtFactory(left__name="foo")]),
            HorizontalExecutionFactory(
                body=[
                    assignment_1 := AssignStmtFactory(
                        left__name="bar", right__name="foo", right__offset__i=1
                    )
                ]
            ),
            HorizontalExecutionFactory(body=[assignment_2 := AssignStmtFactory(right__name="bar")]),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    assert len(transformed.horizontal_executions) == 2
    assert transformed.horizontal_executions[0].body == [assignment_0]
    assert transformed.horizontal_executions[1].body == [assignment_1, assignment_2]


def test_write_after_read_with_offset():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="foo", right__offset__i=1)]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="foo")]),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    for result, reference in zip(transformed.horizontal_executions, testee.horizontal_executions):
        assert result.body == reference.body


def test_nonzero_extent_merging():
    testee = VerticalLoopSectionFactory(
        horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="foo")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(right__name="foo", right__offset__j=1)]
            ),
        ]
    )
    transformed = (
        optimize_horizontal_executions(
            StencilFactory(vertical_loops__0__sections__0=testee),
            GraphMerging,
        )
        .vertical_loops[0]
        .sections[0]
    )
    assert len(transformed.horizontal_executions) == 1
    assert transformed.horizontal_executions[0].body == sum(
        (he.body for he in testee.horizontal_executions), []
    )
