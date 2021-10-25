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

from eve import codegen
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin
from gt4py.backend.gt_backends import (
    GTCUDAPyModuleGenerator,
    cuda_is_compatible_layout,
    cuda_is_compatible_type,
    make_cuda_layout_map,
)
from gt4py.backend.gtc_backend.common import bindings_main_template, pybuffer_to_sid
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.cuir import cuir, cuir_codegen, extent_analysis, kernel_fusion, oir_to_cuir
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_pipeline import OirPipeline


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCCudaExtGenerator:
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
        pass_order = self.backend.builder.options.backend_opts.get("pass_order", {})
        oir_pipeline = OirPipeline(gtir_to_oir.GTIRToOIR().visit(gtir), pass_order)
        oir = oir_pipeline.full(skip=[NoFieldAccessPruning])
        cuir = oir_to_cuir.OIRToCUIR().visit(oir)
        cuir = kernel_fusion.FuseKernels().visit(cuir)
        cuir = extent_analysis.ComputeExtents().visit(cuir)
        cuir = extent_analysis.CacheExtents().visit(cuir)
        format_source = self.backend.builder.options.format_source
        implementation = cuir_codegen.CUIRCodegen.apply(cuir, format_source=format_source)
        bindings = GTCCudaBindingsCodegen.apply(
            cuir, module_name=self.module_name, backend=self.backend, format_source=format_source
        )
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cu": bindings},
        }


class GTCCudaBindingsCodegen(codegen.TemplatedGenerator):
    def __init__(self, backend):
        self.backend = backend
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
                return pybuffer_to_sid(
                    name=node.name,
                    ctype=self.visit(node.dtype),
                    domain_dim_flags=node.dimensions,
                    data_ndim=len(node.data_dims),
                    stride_kind_index=self.unique_index(),
                    backend=self.backend,
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

    Program = bindings_main_template()

    @classmethod
    def apply(cls, root, *, module_name="stencil", backend, **kwargs) -> str:
        generated_code = cls(backend).visit(root, module_name=module_name, **kwargs)
        if kwargs.get("format_source", True):
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code


@gt_backend.register
class GTCCudaBackend(BaseGTBackend, CLIBackendMixin):
    """CUDA backend using gtc."""

    name = "gtc:cuda"
    options = {**BaseGTBackend.GT_BACKEND_OPTS, "device_sync": {"versioning": True, "type": bool}}
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
    USE_LEGACY_TOOLCHAIN = False

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
