import pytest

from gtc.passes.oir_optimizations.horizontal_execution_merging import OnTheFlyMerging
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging
from gtc.passes.oir_pipeline import OirPipeline, hash_step

from .oir_utils import StencilFactory


def test_no_skipping():
    pipeline = OirPipeline(StencilFactory())
    pipeline.full()

    steps = tuple(hash_step(i) for i in pipeline.steps())

    assert steps in pipeline._cache


def test_skip_one():
    pipeline = OirPipeline(StencilFactory())
    pipeline.full(skip=[AdjacentLoopMerging])

    steps = tuple(hash_step(i) for i in pipeline.steps())
    skipped = tuple(i for i in steps if i != hash_step(AdjacentLoopMerging))
    wrong_skipped = tuple(i for i in steps if i != hash_step(OnTheFlyMerging))

    assert steps not in pipeline._cache
    assert skipped in pipeline._cache
    assert wrong_skipped not in pipeline._cache


def test_default_order():
    pipeline = OirPipeline(StencilFactory())
    assert pipeline.steps() == pipeline.default_steps()


def test_reorder_first_and_second():
    step_order = ("AdjacentLoopMerging", "graph_merge_horizontal_executions")
    pipeline = OirPipeline(StencilFactory(), step_order)
    steps = pipeline.steps()
    assert all([steps[i].__name__ == step_order[i]] for i in range(len(step_order)))


def test_reorder_first_and_last():
    step_order = ("FillFlushToLocalKCaches", "graph_merge_horizontal_executions")
    pipeline = OirPipeline(StencilFactory(), step_order)
    steps = pipeline.steps()
    assert all([steps[i].__name__ == step_order[i]] for i in range(len(step_order)))


def test_reorder_before_to_after():
    step_order = {"LocalTemporariesToScalars": 7}
    pipeline = OirPipeline(StencilFactory(), step_order)
    steps = pipeline.steps()
    assert all([steps[index - 1].__name__ == name for name, index in step_order.items()])


def test_reorder_after_to_before():
    step_order = {"KCacheDetection": 4}
    pipeline = OirPipeline(StencilFactory(), step_order)
    steps = pipeline.steps()
    assert all([steps[index].__name__ == name for name, index in step_order.items()])


def test_remove_step():
    step_order = {"LocalTemporariesToScalars": -1}
    pipeline = OirPipeline(StencilFactory(), step_order)
    steps = pipeline.steps()
    assert not any([step.__name__ == list(step_order.keys())[0] for step in steps])


def test_invalid_step():
    step_order = ("FastestOptimizationEver",)
    pipeline = OirPipeline(StencilFactory(), step_order)
    with pytest.raises(RuntimeError, match="Unknown OIR step name"):
        pipeline.steps()
