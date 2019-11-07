# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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


# import pytest
# import itertools
# import numpy as np
#
# import hypothesis as hyp
# import hypothesis.strategies as hyp_st
#
# from gt4py import backend as gt_backend
# from gt4py import storage as gt_store
# from .iir_stencil_definitions import REGISTRY as iir_registry
# from .utils import generate_test_module
#
# from .utils import id_version
#
#
# @pytest.mark.parametrize(
#     ["name", "backend"],
#     itertools.product(
#         iir_registry.names,
#         [
#             gt_backend.from_name(name)
#             for name in gt_backend.REGISTRY.names
#             if gt_backend.from_name(name).storage_info["device"] == "cpu"
#         ],
#     ),
# )
# def test_generation_cpu(name, backend, *, id_version):
#     generate_test_module(name, backend, id_version=id_version)
#
#
# @pytest.mark.requires_cudatoolkit
# @pytest.mark.parametrize(
#     ["name", "backend"],
#     itertools.product(
#         iir_registry.names,
#         [
#             gt_backend.from_name(name)
#             for name in gt_backend.REGISTRY.names
#             if gt_backend.from_name(name).storage_info["device"] == "gpu"
#         ],
#     ),
# )
# def test_generation_gpu(name, backend, *, id_version):
#     generate_test_module(name, backend, id_version=id_version)

import itertools

import numpy as np
import pytest

import gt4py as gt
from gt4py import gtscript
from gt4py import backend as gt_backend
from gt4py import storage as gt_storage


from .stencil_definitions import REGISTRY as stencil_definitions
from .stencil_definitions import EXTERNALS_REGISTRY as externals_registry


@pytest.mark.parametrize(
    ["name", "backend"],
    itertools.product(
        stencil_definitions.names,
        [
            name
            for name in gt_backend.REGISTRY.names
            if gt_backend.from_name(name).storage_info["device"] == "cpu"
        ],
    ),
)
def test_generation_cpu(name, backend):
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


# import gt4py as gt
# import gt4py.storage as gt_storage
# import numpy as np
#
# descr = gt_storage.StorageDescriptor(dtype=np.int_, mask=[True, True, True], grid_group="1")
#
#
# backend = "gtmc"
#
#
# @gt.stencil(backend=backend)
# def copy_stencil(in_field: descr, out_field: descr, *, p: float):
#     if p > 0:
#         out_field = in_field[0, 0, 0] + p
#     else:
#         out_field = in_field[0, 0, 0]
#
#
# field_in = gt_storage.ones(
#     descriptor=descr, backend=backend, halo=(0, 0, 0), iteration_domain=(2, 2, 2)
# )
#
# field_out = gt_storage.zeros(
#     descriptor=descr, backend=backend, halo=(0, 0, 0), iteration_domain=(2, 2, 2)
# )
#
# copy_stencil(in_field=field_in, out_field=field_out, p=3.5)
# print(field_out.data)
