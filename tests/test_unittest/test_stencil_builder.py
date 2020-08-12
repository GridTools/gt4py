import numpy

from gt4py.gtscript import PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder
from gt4py.stencil_object import StencilObject


# type ignores in stencils are because mypy does not yet
# deal with gtscript types well


def simple_stencil(field: Field[float]):  # type: ignore
    from __externals__ import a

    with computation(PARALLEL), interval(...):  # type: ignore
        field += a  # type: ignore


def assign_bool_float(f64_field: Field[numpy.float64], bool_field: Field[numpy.bool]):  # type: ignore  # noqa
    with computation(PARALLEL), interval(...):  # type: ignore
        bool_field = f64_field >= 0  # type: ignore  # noqa


def test_setters():
    # should set some build data (internal cache)
    builder = StencilBuilder(simple_stencil).with_externals({"a": 1.0}).with_backend_data({"b": 2})
    assert builder.externals == {"a": 1.0}
    assert builder.backend_data == {"b": 2}
    version = builder.stencil_id.version
    assert version

    # should reset build data, stencil_id particularly should be recomputed
    builder.with_backend("numpy")
    assert builder.is_build_data_empty
    assert builder.externals == {"a": 1.0}
    assert builder.backend_data == {}
    assert builder.stencil_id.version
    assert builder.stencil_id.version == version

    # should change the stencil version
    builder.with_externals({"a": 2.0})
    assert builder.stencil_id.version != version

    # should reset build data, stencil_id should be recomputed according to
    # new caching strategy
    builder.with_backend_data({"b": 2.0}).with_caching("nocaching")
    assert builder.is_build_data_empty
    assert not builder.stencil_id.version

    # should reset build data
    builder.with_backend_data({"b": 2}).with_externals({"a": 3.0})
    assert builder.is_build_data_empty
    assert builder.externals == {"a": 3.0}


def test_usage_numpy_caching():
    builder = (
        StencilBuilder(simple_stencil)
        .with_backend("numpy")
        .with_externals({"a": 1.0})
        .with_options(
            name=simple_stencil.__name__, module=simple_stencil.__module__, rebuild=False
        )
    )

    # second load should use caching
    builder.backend.generate()
    stencil_cls = builder.backend.load()
    stencil_cls2 = builder.backend.load()
    assert stencil_cls
    assert isinstance(stencil_cls(), StencilObject)
    assert stencil_cls._gt_id_ == stencil_cls2._gt_id_

    # loading without regenerating should fail
    # regenerating should create a different stencil class
    stencil_cls3 = builder.with_externals({"a": 2.0}).backend.load()
    assert stencil_cls3 is None
    builder.backend.generate()
    stencil_cls3 = builder.backend.load()
    assert stencil_cls._gt_id_ != stencil_cls3._gt_id_


def test_usage_numpy_nocaching(tmp_path):
    builder = (
        StencilBuilder(simple_stencil)
        .with_backend("numpy")
        .with_externals({"a": 1.0})
        .with_caching("nocaching", output_path=tmp_path)
        .with_options(name="simple_stencil", module="")
    )

    computation_src = builder.backend.generate_computation()
    assert "simple_stencil.py" in computation_src

    builder.backend.generate()
    assert tmp_path.joinpath("simple_stencil", "simple_stencil.py").exists(), list(
        tmp_path.iterdir()
    )


def test_regression_run_analysis_twice(tmp_path):
    builder = (
        StencilBuilder(assign_bool_float)
        .with_backend("numpy")
        .with_externals({"a": 1.0})
        .with_caching("nocaching", output_path=tmp_path)
        .with_options(name="simple_stencil", module="", rebuild=True)
    )

    # property caching should not reevaluate the analysis pipeline as a side effect.
    ir = builder.implementation_ir
    # this raises an error if the analysis pipeline is reevaluated:
    assert ir is builder.implementation_ir
