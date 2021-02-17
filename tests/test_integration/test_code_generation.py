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

import itertools

import numpy as np
import pytest

import gt4py as gt
from gt4py import backend as gt_backend
from gt4py import gtscript
from gt4py import storage as gt_storage
from gt4py.gtscript import __INLINED, BACKWARD, FORWARD, PARALLEL, computation, interval

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS, OLD_BACKENDS
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
                default_origin=(10, 10, 5),
            )
        else:
            args[k] = v(1.5)
    # vertical domain size >= 16 required for test_large_k_interval
    stencil(**args, origin=(10, 10, 5), domain=(3, 3, 16))


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


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_stage_merger_induced_interval_block_reordering(backend):
    field_in = gt_storage.ones(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )
    field_out = gt_storage.zeros(
        dtype=np.float_, backend=backend, shape=(23, 23, 23), default_origin=(0, 0, 0)
    )

    @gtscript.stencil(backend=backend)
    def stencil(field_in: gtscript.Field[np.float_], field_out: gtscript.Field[np.float_]):
        with computation(BACKWARD):
            with interval(-2, -1):  # block 1
                field_out = field_in
            with interval(0, -2):  # block 2
                field_out = field_in
        with computation(BACKWARD):
            with interval(-1, None):  # block 3
                field_out = 2 * field_in
            with interval(0, -1):  # block 4
                field_out = 3 * field_in

    stencil(field_in, field_out)

    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, 0:-1], 3)
    np.testing.assert_allclose(field_out.view(np.ndarray)[:, :, -1], 2)


@pytest.mark.parametrize("backend", OLD_BACKENDS)
def test_lower_dimensional_inputs(backend):
    @gtscript.stencil(backend=backend)
    def stencil(
        field_3d: gtscript.Field[np.float_, gtscript.IJK],
        field_2d: gtscript.Field[np.float_, gtscript.IJ],
        field_1d: gtscript.Field[np.float_, gtscript.K],
    ):
        with computation(FORWARD):
            with interval(0, 1):
                field_d = field_1d[1] + field_3d[0, 1, 0]

        with computation(PARALLEL):
            with interval(0, 1):
                tmp = field_2d[0, 1] + field_1d[1]
                field_3d = tmp[1, 0, 0] + field_1d[1]
            with interval(1, None):
                field_3d = tmp[-1, 0, 0]

    full_shape = (6, 6, 6)
    default_origin = (1, 1, 0)
    dtype = float

    field_3d = gt_storage.ones(backend, default_origin, full_shape, dtype, mask=None)
    assert field_3d.shape == full_shape[:]

    field_2d = gt_storage.ones(
        backend, default_origin[:-1], full_shape[:-1], dtype, mask=[True, True, False]
    )
    assert field_2d.shape == full_shape[:-1]

    field_1d = gt_storage.zeros(
        backend, (default_origin[-1],), (full_shape[-1],), dtype, mask=[False, False, True]
    )
    assert field_1d.shape == (full_shape[-1],)

    stencil(field_3d, field_2d, field_1d, origin=(1, 1, 0))
    stencil(field_3d, field_2d, field_1d)
