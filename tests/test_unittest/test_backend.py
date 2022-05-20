# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Any, Dict

import numpy as np
import pytest

import gt4py.backend as gt_backend
from gt4py.backend import REGISTRY as backend_registry
from gt4py.backend.module_generator import make_args_data_from_gtir
from gt4py.definitions import AccessKind
from gt4py.gtscript import __INLINED, PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder
from gtc import gtir, utils

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS


def stencil_def(
    out: Field[float],  # type: ignore
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
            out = pa * fa  # type: ignore
        elif __INLINED(MODE == 1):
            out = pa * fa + pb * fb  # type: ignore
        else:
            out = pa * fa + pb * fb - pc * fc  # type: ignore  # noqa


field_info_val = {0: ("out", "fa"), 1: ("out", "fa", "fb"), 2: ("out", "fa", "fb", "fc")}
parameter_info_val = {0: ("pa",), 1: ("pa", "pb"), 2: ("pa", "pb", "pc")}
unreferenced_val = {0: ("pb", "fb", "pc", "fc"), 1: ("pc", "fc"), 2: ()}


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_make_args_data_from_gtir(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    args_data = make_args_data_from_gtir(builder.gtir_pipeline)

    assert set(args_data.unreferenced) == set(unreferenced_val[mode])

    field_info_from_gtir = {
        (
            p.name,
            np.dtype(p.dtype.name.lower()),
            utils.dimension_flags_to_names(p.dimensions).upper(),
            p.data_dims,
        )
        for p in builder.gtir.params
        if isinstance(p, gtir.FieldDecl)
    }
    field_info_from_args_data = {
        (name, d.dtype, "".join(d.axes), d.data_dims)
        for name, d in args_data.field_info.items()
        if name not in args_data.unreferenced
    }
    assert field_info_from_gtir == field_info_from_args_data

    param_info_from_gtir = {
        (p.name, np.dtype(p.dtype.name.lower()))
        for p in builder.gtir.params
        if isinstance(p, gtir.ScalarDecl)
    }
    param_info_from_args_data = {
        (name, d.dtype)
        for name, d in args_data.parameter_info.items()
        if name not in args_data.unreferenced
    }
    assert param_info_from_gtir == param_info_from_args_data

    for name, field_info in args_data.field_info.items():
        if name == "out":
            access = AccessKind.WRITE
        elif name in field_info_val[mode]:
            access = AccessKind.READ
        else:
            access = AccessKind.NONE
        assert field_info.access == access

    for name, param_info in args_data.parameter_info.items():
        if name in parameter_info_val[mode]:
            access = AccessKind.READ
        else:
            access = AccessKind.NONE
        assert param_info.access == access


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_generate_pre_run(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    args_data = make_args_data_from_gtir(builder.gtir_pipeline)

    module_generator = backend_cls.MODULE_GENERATOR_CLASS()
    module_generator.args_data = args_data
    source = module_generator.generate_pre_run()

    if gt_backend.from_name(backend_name).storage_info["device"] == "cpu":
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
    args_data = make_args_data_from_gtir(builder.gtir_pipeline)

    module_generator = backend_cls.MODULE_GENERATOR_CLASS()
    module_generator.args_data = args_data
    source = module_generator.generate_post_run()

    if gt_backend.from_name(backend_name).storage_info["device"] == "cpu":
        assert source == ""
    else:
        assert source == "out._set_device_modified()"


@pytest.mark.parametrize("backend_name", GPU_BACKENDS)
@pytest.mark.parametrize("mode", (2,))
@pytest.mark.parametrize("device_sync", (True, False))
def test_device_sync_option(backend_name, mode, device_sync):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    builder.options.backend_opts["device_sync"] = device_sync
    args_data = make_args_data_from_gtir(builder.gtir_pipeline)
    module_generator = backend_cls.MODULE_GENERATOR_CLASS()
    source = module_generator(
        args_data,
        builder,
        pyext_module_name=builder.module_name,
        pyext_file_path=str(builder.module_path),
    )

    if device_sync:
        assert "cupy.cuda.Device(0).synchronize()" in source
    else:
        assert "cupy.cuda.Device(0).synchronize()" not in source


@pytest.mark.parametrize("backend_name", GPU_BACKENDS)
def test_device_sync_option_registered(backend_name):
    backend_cls = backend_registry[backend_name]
    backend_opts = backend_cls.options
    option_name = "device_sync"
    assert option_name in backend_opts
    assert backend_opts[option_name]["versioning"]
    assert backend_opts[option_name]["type"] is bool


@pytest.mark.parametrize("rebuild", (True, False))
@pytest.mark.parametrize("backend_name", CPU_BACKENDS)
@pytest.mark.parametrize("mode", (2,))
def test_toolchain_profiling(backend_name: str, mode: int, rebuild: bool):
    build_info: Dict[str, Any] = {}
    builder = (
        StencilBuilder(stencil_def)
        .with_backend(backend_name)
        .with_externals({"MODE": mode})
        .with_options(
            name=stencil_def.__name__,
            module=stencil_def.__module__,
            rebuild=rebuild,
            build_info=build_info,
        )
    )
    stencil_class = builder.build()
    assert stencil_class is not None
    if rebuild:
        assert build_info["parse_time"] > 0.0
        assert build_info["module_time"] > 0.0
        if backend_name.startswith("gt") and not backend_name.endswith("numpy"):
            assert build_info["codegen_time"] > 0.0
            assert build_info["build_time"] > 0.0
    else:
        assert build_info["load_time"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__])
