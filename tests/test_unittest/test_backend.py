import inspect

import pytest

from gt4py.backend import REGISTRY as backend_registry
from gt4py.gtscript import __INLINED, PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, DAWN_CPU_BACKENDS


def stencil_def(
    out: Field[float],  # type: ignore  # noqa
    pa: float,
    fa: Field[float],  # type: ignore
    pb: float = None,
    fb: Field[float] = None,  # type: ignore
    pc: float = None,
    fc: Field[float] = None,  # type: ignore
):
    from __externals__ import MODE

    with computation(PARALLEL), interval(...):
        if __INLINED(MODE == 0):
            out = pa * fa  # type: ignore  # noqa
        elif __INLINED(MODE == 1):
            out = pa * fa + pb * fb  # type: ignore  # noqa
        else:
            out = pa * fa + pb * fb - pc * fc  # type: ignore  # noqa


field_info_val = {0: ("out", "fa"), 1: ("out", "fa", "fb"), 2: ("out", "fa", "fb", "fc")}
parameter_info_val = {0: ("pa",), 1: ("pa", "pb"), 2: ("pa", "pb", "pc")}
unreferenced_val = {0: ("pb", "fb", "pc", "fc"), 1: ("pc", "fc"), 2: ()}


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_make_args_data_from_iir(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    iir = builder.implementation_ir
    args_data = backend_cls.make_args_data_from_iir(iir)

    args_list = set(inspect.signature(stencil_def).parameters.keys())
    args_found = set()

    for key in args_data["field_info"]:
        assert key in args_list
        if key in field_info_val[mode]:
            assert args_data["field_info"][key] is not None
        else:
            assert args_data["field_info"][key] is None
        assert key not in args_found
        args_found.add(key)

    for key in args_data["parameter_info"]:
        assert key in args_list
        if key in parameter_info_val[mode]:
            assert args_data["parameter_info"][key] is not None
        else:
            assert args_data["parameter_info"][key] is None
        assert key not in args_found
        args_found.add(key)

    for key in args_data["unreferenced"]:
        assert key in args_list
        assert key not in field_info_val[mode]
        assert key not in parameter_info_val[mode]
        assert key in unreferenced_val[mode]
        assert key in args_found


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_generate_pre_run(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    iir = builder.implementation_ir
    args_data = backend_cls.make_args_data_from_iir(iir)

    module_generator = backend_cls.MODULE_GENERATOR_CLASS()
    module_generator.args_data = args_data
    source = module_generator.generate_pre_run()

    if backend_name in CPU_BACKENDS:
        if backend_name not in DAWN_CPU_BACKENDS:
            assert source == ""
    else:
        for key in field_info_val[mode]:
            assert f"{key}.host_to_device()" in source
        for key in unreferenced_val[mode]:
            assert f"{key}.host_to_device()" not in source


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_generate_post_run(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    iir = builder.implementation_ir
    args_data = backend_cls.make_args_data_from_iir(iir)

    module_generator = backend_cls.MODULE_GENERATOR_CLASS()
    module_generator.args_data = args_data
    source = module_generator.generate_post_run()

    if backend_name in CPU_BACKENDS:
        assert source == ""
    else:
        assert source == "out._set_device_modified()"


if __name__ == "__main__":
    pytest.main([__file__])
