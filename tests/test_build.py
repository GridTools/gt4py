"""Test the backend-agnostic build system."""
import pathlib
from copy import deepcopy

import pytest

import gt4py
from gt4py.build import (
    LazyStencil,
    BuildContext,
    BeginStage,
    IRStage,
    IIRStage,
    SourceStage,
    BindingsStage,
    CompileBindingsStage,
)
from gt4py.gtscript import Field, computation, interval, PARALLEL


def has_cupy():
    try:
        import cupy

        return True
    except ImportError:
        return False


def copy_stencil_definition(out_f: Field[float], in_f: Field[float]):
    """Copy input into output."""
    with computation(PARALLEL), interval(...):
        out_f = in_f


def wrong_syntax_stencil_definition(out_f: Field[float], in_f: Field[float]):
    """Contains a GTScript specific syntax error."""
    with computation(PARALLEL), interval(...):
        out_f += in_f


@pytest.fixture(params=["gtscript"])
def frontend(request):
    yield gt4py.frontend.from_name(request.param)


@pytest.fixture(
    params=[
        "debug",
        "numpy",
        "gtx86",
        "gtmc",
        pytest.param(
            "gtcuda", marks=pytest.mark.skipif(not has_cupy(), reason="cupy not installed")
        ),
    ]
)
def backend_name(request):
    yield request.param


@pytest.fixture
def backend(backend_name):
    yield gt4py.backend.from_name(backend_name)


def test_build_context():
    """Test construction and validation."""
    ctx = BuildContext(copy_stencil_definition)
    ctx.validate()

    assert ctx["name"] == "copy_stencil_definition"
    assert ctx["module"] == "tests.test_build"
    assert ctx["externals"] == {}
    assert ctx["qualified_name"] == "tests.test_build.copy_stencil_definition"
    assert ctx["build_info"] == {}
    assert isinstance(ctx["options"], gt4py.definitions.BuildOptions)

    ctx = BuildContext(copy_stencil_definition, name="copy_stencil")
    ctx.validate()

    assert ctx["qualified_name"] == "tests.test_build.copy_stencil"


def test_lazy_stencil():
    """Test lazy stencil construction."""
    ctx = BuildContext(
        copy_stencil_definition, name="copy_stencil", backend="debug", frontend="gtscript"
    )
    lazy_s = LazyStencil(ctx)

    assert lazy_s.backend.name == "debug"


def test_lazy_syntax_check(frontend, backend):
    """Test syntax checking."""
    lazy_pass = LazyStencil(
        BuildContext(copy_stencil_definition, frontend=frontend, backend=backend)
    )
    lazy_fail = LazyStencil(
        BuildContext(wrong_syntax_stencil_definition, frontend=frontend, backend=backend)
    )
    lazy_pass.check_syntax()
    with pytest.raises(ValueError):
        lazy_fail.check_syntax()


def test_lazy_call(frontend, backend):
    """Test that the lazy stencil is callable like the compiled stencil object."""
    import numpy

    a = gt4py.storage.from_array(
        numpy.array([[[1.0]]]), default_origin=(0, 0, 0), backend=backend.name
    )
    b = gt4py.storage.from_array(
        numpy.array([[[0.0]]]), default_origin=(0, 0, 0), backend=backend.name
    )
    lazy_s = LazyStencil(BuildContext(copy_stencil_definition, frontend=frontend, backend=backend))
    lazy_s(b, a)
    assert b[0, 0, 0] == 1.0


def test_lazy_begin_build(frontend, backend):
    """Test that the lazy stencil returns a BeginStage on begin_build."""
    lazy_s = LazyStencil(BuildContext(copy_stencil_definition, frontend=frontend, backend=backend))
    assert isinstance(lazy_s.begin_build(), BeginStage)


def test_begin_stage(frontend, backend):
    """Begin stage: no requirements and no changes to ctx."""
    ctx = BuildContext(copy_stencil_definition)
    stage = BeginStage(deepcopy(ctx))
    stage.make()
    assert ctx == stage.ctx


def test_ir_stage(frontend, backend):
    """IR stage: test ctx requirements and changes."""
    ctx = BuildContext(copy_stencil_definition)
    ir_stage = IRStage(deepcopy(ctx))
    with pytest.raises(KeyError):
        ir_stage.make()

    ir_stage.ctx["backend"] = backend
    ir_stage.ctx["frontend"] = frontend
    ctx = deepcopy(ir_stage.ctx)
    ir_stage.make()

    assert set(ctx.keys()) != set(ir_stage.ctx.keys())
    assert set(ctx.keys()).issubset(set(ir_stage.ctx.keys()))
    assert ir_stage.ctx["id"]
    assert ir_stage.ctx["ir"]

    ctx = deepcopy(ir_stage.ctx)
    ir_stage.make()
    assert ctx == ir_stage.ctx


def test_iir_stage(frontend, backend):
    """IIR stage: test ctx requirements and changes."""
    ctx = BuildContext(copy_stencil_definition, backend=backend, frontend=frontend)

    iir_stage = IIRStage(deepcopy(ctx))
    with pytest.raises(KeyError):
        iir_stage.make()

    ir_stage = IRStage(deepcopy(ctx)).make()
    ctx["ir"] = ir_stage.ctx["ir"]
    iir_stage = IIRStage(deepcopy(ctx))
    iir_stage.make()

    assert iir_stage.ctx["iir"]

    ctx = deepcopy(iir_stage.ctx)
    iir_stage.make()
    assert ctx == iir_stage.ctx


def test_source_stage(frontend, backend):
    """Test the source build stage."""
    ctx = BuildContext(copy_stencil_definition, backend=backend, frontend=frontend)

    ir_stage = IRStage(deepcopy(ctx)).make()

    source_stage = SourceStage(deepcopy(ir_stage.ctx))
    with pytest.raises(KeyError):
        source_stage.make()

    iir_stage = IIRStage(deepcopy(ir_stage.ctx)).make()

    ctx["iir"] = iir_stage.ctx["iir"]
    if not backend.BINDINGS_LANGUAGES:
        ctx["id"] = iir_stage.ctx["id"]
    source_stage = SourceStage(deepcopy(ctx)).make()
    assert source_stage.ctx["src"]

    ctx = deepcopy(source_stage.ctx)
    source_stage.make()
    assert ctx == source_stage.ctx


@pytest.mark.parametrize(
    "backend_name",
    [
        "gtx86",
        "gtmc",
        pytest.param(
            "gtcuda", marks=pytest.mark.skipif(not has_cupy(), reason="cupy not installed")
        ),
    ],
)
def test_bindings_stage(frontend, backend, tmp_path):
    """Test the bindings stage."""
    ctx = BuildContext(
        copy_stencil_definition, backend=backend, frontend=frontend, bindings=["python"]
    )

    ir_stage = IRStage(deepcopy(ctx)).make()
    iir_stage = IIRStage(deepcopy(ir_stage.ctx)).make()

    ctx["id"] = ir_stage.ctx["id"]
    ctx["iir"] = iir_stage.ctx["iir"]
    ctx["pyext_file_path_final"] = str(tmp_path / "pyext_final")

    bindings_stage = BindingsStage(ctx).make()
    assert ctx["bindings_src"]

    ctx = deepcopy(ctx)
    bindings_stage.make()
    assert ctx == bindings_stage.ctx


@pytest.mark.parametrize(
    "backend_name",
    [
        "gtx86",
        "gtmc",
        pytest.param(
            "gtcuda", marks=pytest.mark.skipif(not has_cupy(), reason="cupy not installed")
        ),
    ],
)
def test_compile_bindings_stage(frontend, backend, tmp_path):
    ctx = BuildContext(
        copy_stencil_definition,
        backend=backend,
        frontend=frontend,
        bindings=["python"],
        compile_bindings=True,
    )
    ctx["pyext_file_path_final"] = str(tmp_path / "_copy_stencil_pyext.so")

    ir_stage = IRStage(deepcopy(ctx)).make()
    iir_stage = IIRStage(deepcopy(ir_stage.ctx)).make()
    source_stage = SourceStage(deepcopy(iir_stage.ctx)).make()
    bindings_stage = BindingsStage(deepcopy(source_stage.ctx)).make()

    ctx["id"] = ir_stage.ctx["id"]
    ctx["iir"] = iir_stage.ctx["iir"]
    ctx["src"] = source_stage.ctx["src"]
    ctx["bindings_src"] = bindings_stage.ctx["bindings_src"]
    ctx["pyext_module_name"] = bindings_stage.ctx["pyext_module_name"]
    ctx["pyext_module_path"] = tmp_path

    files = []
    for source_f, source_c in ctx["src"].items():
        source_path = tmp_path / source_f
        source_path.write_text(source_c)
        if not source_f.endswith(".hpp"):
            files.append(str(source_path))

    for source_f, source_c in ctx["bindings_src"]["python"].items():
        source_path = tmp_path / source_f
        source_path.write_text(source_c)
        if not source_f.endswith(".py"):
            files.append(str(source_path))

    ctx["bindings_src_files"] = {"python": files}
    comp_bind_stage = CompileBindingsStage(ctx).make()
    assert pathlib.Path(ctx["pyext_file_path"]).exists()
    assert (
        pathlib.Path(ctx["pyext_file_path_final"]).joinpath(ctx["pyext_module_name"] + ".so")
        == ctx["pyext_file_path"]
    )

    ctx = deepcopy(ctx)
    comp_bind_stage.make()
    assert ctx == comp_bind_stage.ctx
