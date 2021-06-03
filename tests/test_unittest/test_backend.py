# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

import inspect

import pytest

import gt4py.backend as gt_backend
from gt4py.backend import REGISTRY as backend_registry
from gt4py.backend.module_generator import make_args_data_from_gtir, make_args_data_from_iir
from gt4py.gtscript import __INLINED, PARALLEL, Field, computation, interval
from gt4py.stencil_builder import StencilBuilder

from ..definitions import ALL_BACKENDS, GPU_BACKENDS


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
def test_make_args_data_from_iir(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    iir = builder.implementation_ir
    args_data = make_args_data_from_iir(iir)

    args_list = set(inspect.signature(stencil_def).parameters.keys())
    args_found = set()

    for key in args_data.field_info:
        assert key in args_list
        if key in field_info_val[mode]:
            assert args_data.field_info[key] is not None
        else:
            assert args_data.field_info[key] is None
        assert key not in args_found
        args_found.add(key)

    for key in args_data.parameter_info:
        assert key in args_list
        if key in parameter_info_val[mode]:
            assert args_data.parameter_info[key] is not None
        else:
            assert args_data.parameter_info[key] is None
        assert key not in args_found
        args_found.add(key)

    for key in args_data.unreferenced:
        assert key in args_list
        assert key not in field_info_val[mode]
        assert key not in parameter_info_val[mode]
        assert key in unreferenced_val[mode]
        assert key in args_found


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_make_args_data_from_gtir(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    args_data = make_args_data_from_gtir(builder.gtir_pipeline)
    iir_args_data = make_args_data_from_iir(builder.implementation_ir)

    assert args_data.field_info == iir_args_data.field_info
    assert args_data.parameter_info == iir_args_data.parameter_info
    assert args_data.unreferenced == iir_args_data.unreferenced


@pytest.mark.parametrize("backend_name", ALL_BACKENDS)
@pytest.mark.parametrize("mode", (0, 1, 2))
def test_generate_pre_run(backend_name, mode):
    backend_cls = backend_registry[backend_name]
    builder = StencilBuilder(stencil_def, backend=backend_cls).with_externals({"MODE": mode})
    iir = builder.implementation_ir
    args_data = make_args_data_from_iir(iir)

    module_generator = backend_cls.MODULE_GENERATOR_CLASS()
    module_generator.args_data = args_data
    source = module_generator.generate_pre_run()

    if gt_backend.from_name(backend_name).storage_info["device"] == "cpu":
        if "dawn:" not in backend_name:
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
    args_data = make_args_data_from_iir(iir)

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
    if backend_cls.USE_LEGACY_TOOLCHAIN:
        args_data = make_args_data_from_iir(builder.implementation_ir)
    else:
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


if __name__ == "__main__":
    pytest.main([__file__])
