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
    gtcpu_is_compatible_type,
    make_cuda_layout_map,
    make_mc_layout_map,
    make_x86_layout_map,
    mc_is_compatible_layout,
    x86_is_compatible_layout,
)
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.gtcpp import gtcpp, gtcpp_codegen, oir_to_gtcpp
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging, OnTheFlyMerging
from gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCGTExtGenerator:
    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
        oir = gtir_to_oir.GTIRToOIR().visit(gtir)
        oir = self._optimize_oir(oir)
        gtcpp = oir_to_gtcpp.OIRToGTCpp().visit(oir)
        implementation = gtcpp_codegen.GTCppCodegen.apply(gtcpp, gt_backend_t=self.gt_backend_t)
        bindings = GTCppBindingsCodegen.apply(
            gtcpp, module_name=self.module_name, gt_backend_t=self.gt_backend_t
        )
        bindings_ext = ".cu" if self.gt_backend_t == "gpu" else ".cpp"
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings" + bindings_ext: bindings},
        }

    def _optimize_oir(self, oir):
        oir = GreedyMerging().visit(oir)
        oir = AdjacentLoopMerging().visit(oir)
        oir = LocalTemporariesToScalars().visit(oir)
        oir = WriteBeforeReadTemporariesToScalars().visit(oir)
        oir = OnTheFlyMerging().visit(oir)
        oir = MaskStmtMerging().visit(oir)
        oir = NoFieldAccessPruning().visit(oir)
        oir = IJCacheDetection().visit(oir)
        oir = KCacheDetection().visit(oir)
        oir = PruneKCacheFills().visit(oir)
        oir = PruneKCacheFlushes().visit(oir)
        return oir


class GTCppBindingsCodegen(codegen.TemplatedGenerator):
    def __init__(self):
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    def visit_DataType(self, dtype: DataType, **kwargs):
        return gtcpp_codegen.GTCppCodegen().visit_DataType(dtype)

    def visit_FieldDecl(self, node: gtcpp.FieldDecl, **kwargs):
        assert "gt_backend_t" in kwargs
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
                sid_def = """gt::as_{sid_type}<{dtype}, {sid_ndim},
                    gt::integral_constant<int, {unique_index}>>({name})""".format(
                    sid_type="cuda_sid" if kwargs["gt_backend_t"] == "gpu" else "sid",
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

    def visit_GlobalParamDecl(self, node: gtcpp.GlobalParamDecl, **kwargs):
        if "external_arg" in kwargs:
            if kwargs["external_arg"]:
                return "{dtype} {name}".format(name=node.name, dtype=self.visit(node.dtype))
            else:
                return "gridtools::stencil::make_global_parameter({name})".format(name=node.name)

    def visit_Program(self, node: gtcpp.Program, **kwargs):
        assert "module_name" in kwargs
        entry_params = self.visit(node.parameters, external_arg=True, **kwargs)
        sid_params = self.visit(node.parameters, external_arg=False, **kwargs)
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

    @classmethod
    def apply(cls, root, *, module_name="stencil", **kwargs) -> str:
        generated_code = cls().visit(root, module_name=module_name, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


class GTCGTBaseBackend(BaseGTBackend, CLIBackendMixin):
    options = BaseGTBackend.GT_BACKEND_OPTS
    PYEXT_GENERATOR_CLASS = GTCGTExtGenerator  # type: ignore
    USE_LEGACY_TOOLCHAIN = False

    def _generate_extension(self, uses_cuda: bool) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=uses_cuda)

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


@gt_backend.register
class GTCGTCpuIfirstBackend(GTCGTBaseBackend):
    """GridTools python backend using gtc."""

    name = "gtc:gt:cpu_ifirst"
    GT_BACKEND_T = "cpu_ifirst"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 8,
        "device": "cpu",
        "layout_map": make_mc_layout_map,
        "is_compatible_layout": mc_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=False)


@gt_backend.register
class GTCGTCpuKfirstBackend(GTCGTBaseBackend):
    """GridTools python backend using gtc."""

    name = "gtc:gt:cpu_kfirst"
    GT_BACKEND_T = "cpu_kfirst"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=False)


@gt_backend.register
class GTCGTGpuBackend(GTCGTBaseBackend):
    """GridTools python backend using gtc."""

    MODULE_GENERATOR_CLASS = GTCUDAPyModuleGenerator
    name = "gtc:gt:gpu"
    GT_BACKEND_T = "gpu"
    languages = {"computation": "cuda", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": make_cuda_layout_map,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=True)
