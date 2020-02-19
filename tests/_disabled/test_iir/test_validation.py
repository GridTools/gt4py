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

import numpy as np
import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st

import gt4py.backend as gt_backend
import gt4py.storage as gt_store
from .iir_stencil_definitions import make_test_vertical_regions
from .utils import id_version  # import fixture used by pytest


@pytest.mark.parametrize(
    ["backend"],
    [
        (gt_backend.from_name(name),)
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "cpu"
    ],
)
def test_vertical_regions_cpu(backend, id_version):
    out, ref = compile_and_run_vertical_regions(backend, id_version)
    np.testing.assert_equal(out, ref)


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["backend"],
    [
        (gt_backend.from_name(name),)
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "gpu"
    ],
)
def test_vertical_regions_gpu(backend, id_version):
    out, ref = compile_and_run_vertical_regions(backend, id_version)
    import cupy as cp

    np.testing.assert_equal(cp.asnumpy(out), ref)


def compile_and_run_vertical_regions(backend, id_version):
    import gt4py.definitions as gt_defs
    from gt4py import ir as gt_ir
    from gt4py.definitions import StencilID

    iir = make_test_vertical_regions()

    module_name = "_test_module." + "test_vertical_regions"
    stencil_name = "test_vertical_regions_stencil"
    options = gt_defs.BuildOptions(name=stencil_name, module=module_name, rebuild=False)
    stencil_id = StencilID("{}.{}".format(options.module, options.name), id_version)

    stencil_class = backend.load(stencil_id, None, options)
    if stencil_class is None:
        stencil_class = backend.generate(stencil_id, iir, None, options)

    stencil_implementation = stencil_class()

    field_out = gt_store.from_array(
        np.zeros([d + 2 * o for d, o in zip((10, 10, 10), (0, 0, 0))]),
        default_origin=(0, 0, 0),
        shape=(10, 10, 10),
        dtype=np.float_,
        mask=[True, True, True],
        backend=backend.name,
    )
    field_out_ref = np.zeros([d + 2 * o for d, o in zip((10, 10, 10), (0, 0, 0))])
    field_out_ref[:, :, 0] = 1.0
    field_out_ref[:, :, 1:-1] = 2.0
    field_out_ref[:, :, -1:] = 3.0

    args = dict(_origin_=dict(out=(0, 0, 0)), _domain_=(10, 10, 10), out=field_out)

    if hasattr(field_out, "host_to_device"):
        field_out.host_to_device()
    stencil_implementation.run(**args, exec_info=None)
    if hasattr(field_out, "device_to_host"):
        field_out.device_to_host(force=True)
    return (field_out, field_out_ref)


@pytest.mark.parametrize(
    ["backend"],
    [
        (gt_backend.from_name(name),)
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "cpu"
        and issubclass(gt_backend.from_name(name), gt_backend.BaseGTBackend)
    ],
)
@hyp.given(
    origin_in=hyp_st.tuples(*(([hyp_st.integers(min_value=1, max_value=10)] * 3))),
    origin_out=hyp_st.tuples(*(([hyp_st.integers(min_value=0, max_value=10)] * 3))),
    domain=hyp_st.tuples(*([hyp_st.integers(min_value=1, max_value=32)] * 3)),
)
def test_iteration_domain_cpu(backend, id_version, origin_in, origin_out, domain):
    out, ref = compile_and_run_average_stencil(backend, id_version, origin_in, origin_out, domain)
    np.testing.assert_equal(out, ref)


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["backend"],
    [
        (gt_backend.from_name(name),)
        for name in gt_backend.REGISTRY.names
        if gt_backend.from_name(name).storage_info["device"] == "gpu"
        and issubclass(gt_backend.from_name(name), gt_backend.BaseGTBackend)
    ],
)
@hyp.given(
    origin_in=hyp_st.tuples(*(([hyp_st.integers(min_value=1, max_value=10)] * 3))),
    origin_out=hyp_st.tuples(*(([hyp_st.integers(min_value=0, max_value=10)] * 3))),
    domain=hyp_st.tuples(*([hyp_st.integers(min_value=1, max_value=32)] * 3)),
)
def test_iteration_domain_gpu(backend, id_version, origin_in, origin_out, domain):
    out, ref = compile_and_run_average_stencil(backend, id_version, origin_in, origin_out, domain)
    import cupy as cp

    np.testing.assert_equal(cp.asnumpy(out), ref)


def compile_and_run_average_stencil(backend, id_version, origin_in, origin_out, domain):
    import gt4py.definitions as gt_defs
    from gt4py import ir as gt_ir
    from gt4py.definitions import StencilID

    def stage(field_in, field_out):
        field_out = (
            field_in[-1, 0, 0]
            + field_in[1, 0, 0]
            + field_in[0, -1, 0]
            + field_in[0, 1, 0]
            + field_in[0, 0, -1]
            + field_in[0, 0, 1]
        ) / 6

    stage_1 = gt_ir.utils.make_stage(
        stage,
        compute_extent=[(1, 1), (1, 1), (1, 1)],
        fields_with_access={
            "field_in": (gt_ir.AccessIntent.READ_ONLY, [(1, 1), (1, 1), (1, 1)]),
            "field_out": (gt_ir.AccessIntent.READ_WRITE, [(0, 0), (0, 0), (0, 0)]),
        },
    )

    multi_stage_1 = gt_ir.utils.make_multi_stage(
        "multi_stage_1", gt_ir.IterationOrder.PARALLEL, [stage_1]
    )

    iir = gt_ir.utils.make_implementation(
        "test_iteration_domain_stencil",
        args_list=["field_in", "field_out"],
        fields_with_description={
            "field_in": dict(is_api=True, extent=[(1, 1), (1, 1), (1, 1)]),
            "field_out": dict(is_api=True, extent=[(0, 0), (0, 0), (0, 0)]),
        },
        parameters_with_type={},
        multi_stages=[multi_stage_1],
    )

    module_name = "_test_module." + "test_iteration_domain"
    stencil_name = "test_iteration_domain_stencil"
    options = gt_defs.BuildOptions(
        name=stencil_name, module=module_name, rebuild=False, backend_opts={"verbose": True}
    )
    stencil_id = StencilID("{}.{}".format(options.module, options.name), id_version)

    stencil_class = backend.load(stencil_id, None, options)
    if stencil_class is None:
        stencil_class = backend.generate(stencil_id, iir, None, options)

    stencil_implementation = stencil_class()

    field_in = gt_store.from_array(
        np.ones([d + 2 * o for d, o in zip(domain, origin_in)]),
        default_origin=origin_in,
        dtype=np.float_,
        mask=[True, True, True],
        backend=backend.name,
    )
    field_out = gt_store.from_array(
        np.zeros([d + 2 * o for d, o in zip(domain, origin_out)]),
        dtype=np.float_,
        mask=[True, True, True],
        default_origin=origin_out,
        backend=backend.name,
    )
    field_out_ref = np.zeros([d + 2 * o for d, o in zip(domain, origin_out)])
    slices_out = tuple([slice(o, o + d, None) for o, d in zip(origin_out, domain)])
    field_out_ref[slices_out] = 1.0

    args = dict(
        _origin_=dict(field_in=origin_in, field_out=origin_out),
        _domain_=domain,
        field_in=field_in,
        field_out=field_out,
    )
    if hasattr(field_in, "host_to_device"):
        field_in.host_to_device()
    if hasattr(field_out, "host_to_device"):
        field_out.host_to_device()
    stencil_implementation.run(**args, exec_info=None)
    if hasattr(field_in, "device_to_host"):
        field_in.device_to_host(force=True)
    if hasattr(field_out, "device_to_host"):
        field_out.device_to_host(force=True)
    return (field_out.data, field_out_ref)
