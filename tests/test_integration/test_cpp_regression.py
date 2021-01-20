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
import os

import hypothesis as hyp
import hypothesis.strategies as hyp_st
import numpy as np
import pytest

import gt4py.backend as gt_backend
import gt4py.storage as gt_store

from ..definitions import ALL_BACKENDS, CPU_BACKENDS, GPU_BACKENDS, INTERNAL_BACKENDS, id_version
from ..reference_cpp_regression import reference_module
from .utils import id_version  # import fixture used by pytest
from .utils import generate_test_module


INTERNAL_CPU_BACKENDS = list(set(CPU_BACKENDS) & set(INTERNAL_BACKENDS))
INTERNAL_GPU_BACKENDS = list(set(GPU_BACKENDS) & set(INTERNAL_BACKENDS))

REGISTRY = list()


def register(cpp_regression_test):
    REGISTRY.append(cpp_regression_test)
    return cpp_regression_test


def get_reference(test_name, backend, domain, origins, shapes, masks=None):

    reference_data = reference_module.__dict__[test_name](*domain)

    res = {}
    for k, data in reference_data.items():
        if np.isscalar(data):
            res[k] = np.float_(data)
        else:
            try:

                field = gt_store.storage(
                    data,
                    dtype=np.float_,
                    halo=origins[k],
                    defaults=backend.name,
                )
            except KeyError:
                field = gt_store.storage(
                    data,
                    dtype=np.float_,
                    halo=origins[k[: -len("_reference")]],
                    defaults=backend.name,
                )

            res[k] = field
    return res


@register
@hyp.given(domain=hyp_st.tuples(*([hyp_st.integers(min_value=1, max_value=8)] * 3)))
def run_horizontal_diffusion(backend, id_version, domain):

    validate_field_names = ["out_field"]
    origins = {"in_field": (2, 2, 0), "out_field": (0, 0, 0), "coeff": (0, 0, 0)}
    shapes = {
        name: tuple(domain[i] + 2 * origin[i] for i in range(3)) for name, origin in origins.items()
    }
    name = "horizontal_diffusion"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = generate_test_module(
        "horizontal_diffusion", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        arg_fields[k].host_to_device()
    testmodule.run(
        **arg_fields,
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        arg_fields[k].device_to_host(
            force=True
        )  # directly called `run`, therefore storages are not marked modified

        np.testing.assert_allclose(
            arg_fields[k].to_numpy(), validate_fields[k + "_reference"].to_numpy()
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
    shapes = {
        name: tuple(domain[i] + 2 * origin[i] for i in range(3)) for name, origin in origins.items()
    }
    name = "tridiagonal_solver"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = generate_test_module(
        "tridiagonal_solver", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        arg_fields[k].host_to_device()
    testmodule.run(
        **arg_fields,
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        print(type(arg_fields[k]))
        arg_fields[k].device_to_host(
            force=True
        )  # directly called `run`, therefore storages are not marked modified
        np.testing.assert_allclose(
            arg_fields[k].to_numpy(), validate_fields[k + "_reference"].to_numpy()
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
    testmodule.run(
        **arg_fields,
        _domain_=domain,
        _origin_=origins,
        # _origin_={
        #    k: [oo[0] if isinstance(oo, tuple) else oo for oo in o] for k, o in origins.items()
        # },
        exec_info=None,
    )

    for k in validate_field_names:
        arg_fields[k].device_to_host(
            force=True
        )  # directly called `run`, therefore storages are not marked modified
        np.testing.assert_allclose(
            arg_fields[k].to_numpy(), validate_fields[k + "_reference"].to_numpy()
        )


@register
@hyp.given(
    domain=hyp_st.tuples(
        *(
            [hyp_st.integers(min_value=1, max_value=32)] * 2
            + [hyp_st.integers(min_value=16, max_value=32)]
        )
    )
)
def run_large_k_interval(backend, id_version, domain):
    """Test stencils with large static and potentially zero-length intervals."""
    validate_field_names = ["out_field"]
    origins = {"in_field": (0, 0, 0), "out_field": (0, 0, 0)}
    shapes = {
        name: tuple(domain[i] + 2 * origin[i] for i in range(3)) for name, origin in origins.items()
    }
    name = "large_k_interval"

    arg_fields = get_reference(name, backend, domain, origins, shapes)
    validate_fields = {
        name + "_reference": arg_fields.pop(name + "_reference") for name in validate_field_names
    }

    testmodule = generate_test_module(
        "large_k_interval", backend, id_version=id_version, rebuild=False
    )
    for k in arg_fields:
        arg_fields[k].host_to_device()
    testmodule.run(
        **arg_fields,
        _domain_=domain,
        _origin_=origins,
        exec_info=None,
    )

    for k in validate_field_names:
        arg_fields[k].device_to_host(
            force=True
        )  # directly called `run`, therefore storages are not marked modified
        np.testing.assert_allclose(
            arg_fields[k].to_numpy(), validate_fields[k + "_reference"].to_numpy()
        )


@pytest.mark.parametrize(
    ["backend", "function"], itertools.product(INTERNAL_CPU_BACKENDS, REGISTRY)
)
def test_cpp_regression_cpu(backend, id_version, function):
    function(gt_backend.from_name(backend), id_version)


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["backend", "function"], itertools.product(INTERNAL_GPU_BACKENDS, REGISTRY)
)
def test_cpp_regression_gpu(backend, id_version, function):
    function(gt_backend.from_name(backend), id_version)
