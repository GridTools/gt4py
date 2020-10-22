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
from gt4py import gtscript
from gt4py import storage as gt_storage

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS
from .stencil_definitions import PROPERTY_REGISTRY as property_registry
from .stencil_definitions import REGISTRY as stencil_definitions


@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, CPU_BACKENDS)
)
def test_generation_cpu(name, backend):
    stencil_definition = stencil_definitions[name]
    properties = property_registry[name]
    externals, splitters = (properties.get(prop, {}) for prop in ("externals", "splitters"))
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
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3), splitters=splitters)


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["name", "backend"], itertools.product(stencil_definitions.names, GPU_BACKENDS)
)
def test_generation_gpu(name, backend):
    stencil_definition = stencil_definitions[name]
    properties = property_registry[name]
    externals, splitters = (properties.get(prop, {}) for prop in ("externals", "splitters"))
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
    stencil(**args, origin=(10, 10, 10), domain=(3, 3, 3), splitters=splitters)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_temporary_field_declared_in_if(backend):
    @gtscript.stencil(backend=backend)
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


def test_ignore_np_errstate():
    def setup_and_run(backend, **kwargs):
        field_a = gt_storage.zeros(
            dtype=np.float_,
            backend=backend,
            shape=(3, 3, 1),
            default_origin=(0, 0, 0),
        )

        @gtscript.stencil(backend=backend, **kwargs)
        def divide_by_zero(field_a: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                field_a = 1.0 / field_a

        divide_by_zero(field_a)

    # Usual behavior: with the numpy backend there is no error
    setup_and_run(backend="numpy")

    # Expect warning with debug or numpy + ignore_np_errstate=False
    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        setup_and_run(backend="debug")

    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        setup_and_run(backend="numpy", ignore_np_errstate=False)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stencil_without_effect(backend):
    def definition1(field_in: gtscript.Field[np.float_]):
        with computation(PARALLEL), interval(...):
            tmp = 0.0

    def definition2(f_in: gtscript.Field[np.float_]):
        from __externals__ import flag
        from __gtscript__ import __INLINE

        with computation(PARALLEL), interval(...):
            if __INLINED(flag):
                B = f_in

    stencil1 = gtscript.stencil(backend, definition1)
    stencil2 = gtscript.stencil(backend, definition2, externals={"flag": False})

    field_in = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )

    # test with explicit domain specified
    stencil1(field_in, domain=(3, 3, 3))
    stencil2(field_in, domain=(3, 3, 3))

    # test without domain specified
    with pytest.raises(ValueError):
        stencil1(field_in)
