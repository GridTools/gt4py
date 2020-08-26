import pytest

import gt4py
from gt4py.gtscript import PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder


@pytest.fixture(
    params=[
        pytest.param(
            name,
            marks=pytest.mark.skipif(
                name.startswith("dawn"), reason="dawn backends not yet supported"
            ),
        )
        for name in gt4py.backend.REGISTRY.keys()
    ]
)
def backend(request):
    """Parametrize by backend name."""
    yield gt4py.backend.from_name(request.param)


# mypy gets confused by gtscript
def init_1(input_field: Field[float]):  # type: ignore
    """Implement simple stencil."""
    with computation(PARALLEL), interval(...):  # type: ignore
        input_field = 1  # noqa - unused var is in/out field


def test_generate_computation(backend, tmp_path):
    """
    Test the :py:meth:`gt4py.backend.CLIBackendMixin.generate_computation` method.

    Assumption:

        * All registered backends support the `generate_computation` api.

    Actions:

        Generate the computation source code for a simple stencil.

    Outcome:

        The computation source code and file hierarchy specification is returned.

    """
    builder = StencilBuilder(init_1, backend=backend).with_caching(
        "nocaching", output_path=tmp_path / __name__ / "generate_computation"
    )
    result = builder.backend.generate_computation()

    py_result = backend.languages["computation"] == "python" and "init_1.py" in result
    gt_result = (
        backend.languages["computation"] in {"c++", "cuda"}
        and "init_1_src" in result
        and "computation.hpp" in result["init_1_src"]
        and "bindings.cpp" not in result["init_1_src"]
    )
    assert py_result or gt_result


def test_generate_bindings(backend, tmp_path):
    """
    Test :py:meth:`gt4py.backend.CLIBackendMixin.generate_bindings.
    """
    builder = StencilBuilder(init_1, backend=backend).with_caching(
        "nocaching", output_path=tmp_path / __name__ / "generate_bindings"
    )
    if backend.languages["computation"] == "python":
        with pytest.raises(NotImplementedError):
            result = builder.backend.generate_bindings("python")
    else:
        result = builder.backend.generate_bindings("python")
        assert "init_1_src" in result
        assert "bindings.cpp" in result["init_1_src"]
