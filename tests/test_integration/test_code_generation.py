# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
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

import itertools

import numpy as np
import pytest

import gt4py as gt
from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py import gtscript
from gt4py import stencil_builder as gt_builder
from gt4py import storage as gt_storage

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS
from .stencil_definitions import EXTERNALS_REGISTRY as externals_registry
from .stencil_definitions import REGISTRY as stencil_definitions


@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, CPU_BACKENDS)
)
def test_generation_cpu(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals, rebuild=True)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3))


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, GPU_BACKENDS)
)
def test_generation_gpu(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals)
    args = {}
    for k, v in stencil_definition.__annotations__.items():
        if isinstance(v, gtscript._FieldDescriptor):
            args[k] = gt_storage.ones(
                dtype=v.dtype,
                mask=gtscript.mask_from_axes(v.axes),
                backend=backend,
                shape=(23, 23, 23),
                default_origin=(10, 10, 10),
            )
        else:
            args[k] = v(1.5)
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3))


def test_temporary_field_declared_in_if_raises():

    from gt4py.frontend.gtscript_frontend import GTScriptSymbolError

    with pytest.raises(GTScriptSymbolError):

        @gtscript.stencil(backend="debug")
        def definition(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                if field_a < 0:
                    field_b = -field_a
                else:
                    field_b = field_a
                field_a = field_b


@pytest.mark.requires_gpu
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stage_without_effect(backend):
    @gtscript.stencil(backend=backend)
    def definition(field_a: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            field_c = 0.0


@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, CPU_BACKENDS)
)
def test_cache_info(name, backend):
    stencil_definition = stencil_definitions[name]
    externals = externals_registry[name]
    stencil = gtscript.stencil(backend, stencil_definition, externals=externals)

    builder = gt_builder.StencilBuilder(
        stencil_definition,
        backend=gt_backend.from_name(backend),
        options=gt_definitions.BuildOptions(name=name, module=stencil.options["module"]),
    )
    builder.with_externals(externals)

    assert builder.caching.is_cache_info_available_and_consistent(validate_hash=True)
