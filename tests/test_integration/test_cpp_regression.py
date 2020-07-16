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

import os
import pytest
import numpy as np
import itertools

import gt4py.storage as gt_store
import gt4py.backend as gt_backend

from .utils import id_version  # import fixture used by pytest
from ..reference_cpp_regression import reference_module

from .utils import generate_test_module
import hypothesis as hyp
import hypothesis.strategies as hyp_st

REGISTRY = list()


def register(cpp_regression_test):
    REGISTRY.append(cpp_regression_test)
    return cpp_regression_test


def get_reference(test_name, backend, domain, origins, shapes, dtype=np.float64, masks=None):

    reference_data = reference_module.__dict__[test_name](*domain)

    res = {}
    for k, data in reference_data.items():
        if np.isscalar(data):
            res[k] = dtype(data)
        else:
            try:
                field = gt_store.from_array(
                    data,
                    dtype=dtype,
                    default_origin=origins[k],
                    shape=shapes[k],
                    backend=backend.name,
                )
            except KeyError:
                field = gt_store.from_array(
                    data,
                    dtype=dtype,
                    default_origin=origins[k[: -len("_reference")]],
                    shape=shapes[k[: -len("_reference")]],
                    backend=backend.name,
                )

            res[k] = field
    return res


@register
@hyp.given(domain=hyp_st.tuples(*([hyp_st.integers(min_value=1, max_value=8)] * 3)))
def run_horizontal_diffusion(backend, id_version, domain):

    validate_field_names = ["out_field"]
    origins = {"in_field": (2, 2, 0), "out_field": (0, 0, 0), "coeff": (0, 0, 0)}
    shapes = {k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()}
    name = "horizontal_diffusion"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = generate_test_module(
        "horizontal_diffusion", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()
    testmodule.run(
        **arg_fields,
        # **{k: v.view(np.ndarray) for k, v in arg_fields.items()},
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        if hasattr(arg_fields[k], "synchronize"):
            arg_fields[k].device_to_host(force=True)
        np.testing.assert_allclose(
            arg_fields[k].view(np.ndarray), validate_fields[k + "_reference"].view(np.ndarray)
        )


@register
@hyp.given(
    domain=hyp_st.tuples(
        *(
            [hyp_st.integers(min_value=1, max_value=32)] * 2
            + [hyp_st.integers(min_value=2, max_value=32)]
        )
    )
)
def run_tridiagonal_solver(backend, id_version, domain):

    validate_field_names = ["out"]
    origins = {
        "inf": (0, 0, 0),
        "diag": (0, 0, 0),
        "sup": (0, 0, 0),
        "rhs": (0, 0, 0),
        "out": (0, 0, 0),
    }
    shapes = {k: tuple(domain[i] + 2 * origins[k][i] for i in range(3)) for k in origins.keys()}
    name = "tridiagonal_solver"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = generate_test_module(
        "tridiagonal_solver", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()
    testmodule.run(
        **arg_fields,
        # **{k: v.view(np.ndarray) for k, v in arg_fields.items()},
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        if hasattr(arg_fields[k], "synchronize"):
            arg_fields[k].device_to_host(force=True)
        np.testing.assert_allclose(
            arg_fields[k].view(np.ndarray), validate_fields[k + "_reference"].view(np.ndarray)
        )


@register
@hyp.given(
    domain=hyp_st.tuples(
        *(
            [hyp_st.integers(min_value=1, max_value=32)] * 2
            + [hyp_st.integers(min_value=2, max_value=32)]
        )
    )
)
def run_vertical_advection_dycore(backend, id_version, domain):

    validate_field_names = ["utens_stage"]
    origins = {
        "utens_stage": (0, 0, 0),
        "u_stage": (0, 0, 0),
        "wcon": (0, 0, 0),
        "u_pos": (0, 0, 0),
        "utens": (0, 0, 0),
    }
    shapes = {
        "utens_stage": domain,
        "u_stage": domain,
        "wcon": tuple(d + 1 if i == 0 else d for i, d in enumerate(domain)),
        "u_pos": domain,
        "utens": domain,
    }
    name = "vertical_advection_dycore"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = generate_test_module(
        "vertical_advection_dycore", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        if hasattr(arg_fields[k], "host_to_device"):
            arg_fields[k].host_to_device()
    testmodule.run(**arg_fields, _domain_=domain, _origin_=origins, exec_info=None)

    for k in validate_field_names:
        if hasattr(arg_fields[k], "synchronize"):
            arg_fields[k].device_to_host(force=True)
        np.testing.assert_allclose(
            arg_fields[k].view(np.ndarray), validate_fields[k + "_reference"].view(np.ndarray)
        )


@pytest.mark.parametrize(
    ["backend", "function"],
    itertools.product(
        [
            gt_backend.from_name(name)
            for name in gt_backend.REGISTRY.names
            # for name in ["numpy"]
            if gt_backend.from_name(name).storage_info["device"] == "cpu"
        ],
        REGISTRY,
    ),
)
def test_cpp_regression_cpu(backend, id_version, function):
    function(backend, id_version)


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["backend", "function"],
    itertools.product(
        [
            gt_backend.from_name(name)
            for name in gt_backend.REGISTRY.names
            if gt_backend.from_name(name).storage_info["device"] == "gpu"
        ],
        REGISTRY,
    ),
)
def test_cpp_regression_gpu(backend, id_version, function):
    function(backend, id_version)
