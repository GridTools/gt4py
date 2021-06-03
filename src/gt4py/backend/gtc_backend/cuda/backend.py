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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

import gtc.utils as gtc_utils
from eve import codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin
from gt4py.backend.gt_backends import (
    GTCUDAPyModuleGenerator,
    cuda_is_compatible_layout,
    cuda_is_compatible_type,
    make_cuda_layout_map,
)
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.cuir import cuir, cuir_codegen, extent_analysis, kernel_fusion, oir_to_cuir
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast
from gtc.passes.oir_dace_optimizations import GraphMerging, optimize_horizontal_executions
from gtc.passes.oir_optimizations.caches import (
    FillFlushToLocalKCaches,
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import OnTheFlyMerging
from gtc.passes.oir_optimizations.inlining import MaskInlining
from gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCCudaExtGenerator:
    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.options = options

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtir = DefIRToGTIR.apply(definition_ir)
        gtir_without_unused_params = prune_unused_parameters(gtir)
        dtype_deduced = resolve_dtype(gtir_without_unused_params)
        upcasted = upcast(dtype_deduced)
        oir = gtir_to_oir.GTIRToOIR().visit(upcasted)
        oir = self._optimize_oir(oir)
        cuir = oir_to_cuir.OIRToCUIR().visit(oir)
        cuir = kernel_fusion.FuseKernels().visit(cuir)
        cuir = extent_analysis.ComputeExtents().visit(cuir)
        cuir = extent_analysis.CacheExtents().visit(cuir)
        implementation = cuir_codegen.CUIRCodegen.apply(cuir)
        bindings = GTCCudaBindingsCodegen.apply(cuir, module_name=self.module_name)
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cu": bindings},
        }

    def _optimize_oir(self, oir):
        oir = optimize_horizontal_executions(oir, GraphMerging)
        oir = AdjacentLoopMerging().visit(oir)
        oir = LocalTemporariesToScalars().visit(oir)
        oir = WriteBeforeReadTemporariesToScalars().visit(oir)
        oir = OnTheFlyMerging().visit(oir)
        oir = MaskStmtMerging().visit(oir)
        oir = MaskInlining().visit(oir)
        oir = IJCacheDetection().visit(oir)
        oir = KCacheDetection().visit(oir)
        oir = PruneKCacheFills().visit(oir)
        oir = PruneKCacheFlushes().visit(oir)
        oir = FillFlushToLocalKCaches().visit(oir)
        return oir


class GTCCudaBindingsCodegen(codegen.TemplatedGenerator):
    def __init__(self):
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    def visit_DataType(self, dtype: DataType, **kwargs):
        return cuir_codegen.CUIRCodegen().visit_DataType(dtype)

    def visit_FieldDecl(self, node: cuir.FieldDecl, **kwargs):
        if "external_arg" in kwargs:
            domain_ndim = node.dimensions.count(True)
            data_ndim = len(node.data_dims)
            sid_ndim = domain_ndim + data_ndim
            if kwargs["external_arg"]:
                return "py::buffer {name}, std::array<gt::uint_t,{sid_ndim}> {name}_origin".format(
                    name=node.name,
                    sid_ndim=sid_ndim,
                )
            else:
                sid_def = """gt::as_cuda_sid<{dtype}, {sid_ndim},
                    gt::integral_constant<int, {unique_index}>>({name})""".format(
                    name=node.name,
                    dtype=self.visit(node.dtype),
                    unique_index=self.unique_index(),
                    sid_ndim=sid_ndim,
                )
                if domain_ndim != 3:
                    gt_dims = [
                        f"gt::stencil::dim::{dim}"
                        for dim in gtc_utils.dimension_flags_to_names(node.dimensions)
                    ]
                    if data_ndim:
                        gt_dims += [
                            f"gt::integral_constant<int, {3 + dim}>" for dim in range(data_ndim)
                        ]
                    sid_def = "gt::sid::rename_numbered_dimensions<{gt_dims}>({sid_def})".format(
                        gt_dims=", ".join(gt_dims), sid_def=sid_def
                    )

                return "gt::sid::shift_sid_origin({sid_def}, {name}_origin)".format(
                    sid_def=sid_def,
                    name=node.name,
                )

    def visit_ScalarDecl(self, node: cuir.ScalarDecl, **kwargs):
        if "external_arg" in kwargs:
            if kwargs["external_arg"]:
                return "{dtype} {name}".format(name=node.name, dtype=self.visit(node.dtype))
            else:
                return "gridtools::stencil::make_global_parameter({name})".format(name=node.name)

    def visit_Program(self, node: cuir.Program, **kwargs):
        assert "module_name" in kwargs
        entry_params = self.visit(node.params, external_arg=True, **kwargs)
        sid_params = self.visit(node.params, external_arg=False, **kwargs)
        return self.generic_visit(
            node,
            entry_params=entry_params,
            sid_params=sid_params,
            **kwargs,
        )

    Program = as_mako(
        """
        #include <chrono>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/sid/sid_shift_origin.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include "computation.hpp"
        namespace gt = gridtools;
        namespace py = ::pybind11;
        %if len(entry_params) > 0:
        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", [](std::array<gt::uint_t, 3> domain,
            ${','.join(entry_params)},
            py::object exec_info){
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
        %endif
        """
    )

    @classmethod
    def apply(cls, root, *, module_name="stencil", **kwargs) -> str:
        generated_code = cls().visit(root, module_name=module_name, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


@gt_backend.register
class GTCCudaBackend(BaseGTBackend, CLIBackendMixin):
    """CUDA backend using gtc."""

    name = "gtc:cuda"
    options = BaseGTBackend.GT_BACKEND_OPTS
    languages = {"computation": "cuda", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": make_cuda_layout_map,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }
    PYEXT_GENERATOR_CLASS = GTCCudaExtGenerator  # type: ignore
    MODULE_GENERATOR_CLASS = GTCUDAPyModuleGenerator
    GT_BACKEND_T = "gpu"

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=True)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]

        # TODO(havogt) add bypass if computation has no effect
        pyext_module_name, pyext_file_path = self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )
