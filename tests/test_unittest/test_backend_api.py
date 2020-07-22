import pytest

import gt4py
from gt4py.gtscript import Field, computation, interval, PARALLEL


@pytest.fixture(
    params=[
        pytest.param(
            name,
            marks=pytest.mark.xfail(
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


def test_generate_computation(backend):
    """
    Test the :py:meth:`gt4py.backend.Backend.generate_computation` method.

    Assumption:

        * All registered backends support the `generate_computation` api.

    Actions:

        Generate the computation source code for a simple stencil.

    Outcome:

        The computation source code and file hierarchy specification is returned.

    """
    options = gt4py.definitions.BuildOptions(name=init_1.__name__, module=init_1.__module__)
    frontend = gt4py.frontend.from_name("gtscript")
    init_1_ir = frontend.generate(init_1, externals={}, options=options)
    result = backend.generate_computation(init_1_ir, options=options)

    py_result = backend.languages["computation"] == "python" and "init_1.py" in result
    gt_result = (
        backend.languages["computation"] in {"c++", "cuda"}
        and "init_1_src" in result
        and "computation.hpp" in result["init_1_src"]
    )
    assert py_result or gt_result
