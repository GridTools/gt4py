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


from typing import Tuple

import gtc.utils as gtc_utils
from eve.codegen import MakoTemplate as as_mako


def _get_unit_stride_dim(backend, domain_dim_flags, data_ndim):
    make_layout_map = backend.storage_info["layout_map"]
    layout_map = [
        x for x in make_layout_map(domain_dim_flags + (True,) * data_ndim) if x is not None
    ]
    return layout_map.index(max(layout_map))


def pybuffer_to_sid(
    *,
    name: str,
    ctype: str,
    domain_dim_flags: Tuple[bool, bool, bool],
    data_ndim: int,
    stride_kind_index: int,
    backend,
):
    domain_ndim = domain_dim_flags.count(True)
    sid_ndim = domain_ndim + data_ndim

    as_sid = "as_cuda_sid" if backend.GT_BACKEND_T == "gpu" else "as_sid"

    sid_def = """gt::{as_sid}<{ctype}, {sid_ndim},
        gt::integral_constant<int, {unique_index}>, {unit_stride_dim}>({name})""".format(
        name=name,
        ctype=ctype,
        unique_index=stride_kind_index,
        sid_ndim=sid_ndim,
        as_sid=as_sid,
        unit_stride_dim=_get_unit_stride_dim(backend, domain_dim_flags, data_ndim),
    )
    sid_def = "gt::sid::shift_sid_origin({sid_def}, {name}_origin)".format(
        sid_def=sid_def,
        name=name,
    )
    if domain_ndim != 3:
        gt_dims = [
            f"gt::stencil::dim::{dim}"
            for dim in gtc_utils.dimension_flags_to_names(domain_dim_flags)
        ]
        if data_ndim:
            gt_dims += [f"gt::integral_constant<int, {3 + dim}>" for dim in range(data_ndim)]
        sid_def = "gt::sid::rename_numbered_dimensions<{gt_dims}>({sid_def})".format(
            gt_dims=", ".join(gt_dims), sid_def=sid_def
        )

    return sid_def


def bindings_main_template():
    return as_mako(
        """
        #include <chrono>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <gridtools/stencil/cartesian.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/sid/sid_shift_origin.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include "computation.hpp"
        namespace gt = gridtools;
        namespace py = ::pybind11;
        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", [](
            ${','.join(["std::array<gt::uint_t, 3> domain", *entry_params, 'py::object exec_info'])}
            ){
                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_start_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
                }

                ${name}(domain)(${','.join(sid_params)});

                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_end_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
                }

            }, "Runs the given computation");}
        """
    )
