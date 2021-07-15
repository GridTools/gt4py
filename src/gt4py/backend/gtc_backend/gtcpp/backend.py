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
    gtcpu_is_compatible_type,
    make_cuda_layout_map,
    make_mc_layout_map,
    make_x86_layout_map,
    mc_is_compatible_layout,
    x86_is_compatible_layout,
)
from gt4py.backend.gtc_backend.common import bindings_main_template, pybuffer_to_sid
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.gtcpp import gtcpp, gtcpp_codegen, oir_to_gtcpp
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging
from gtc.passes.oir_pipeline import OirPipeline


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCGTExtGenerator:
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
        oir = OirPipeline(gtir_to_oir.GTIRToOIR().visit(gtir)).full(
            skip=[GreedyMerging, FillFlushToLocalKCaches]
        )
        gtcpp = oir_to_gtcpp.OIRToGTCpp().visit(oir)
        implementation = gtcpp_codegen.GTCppCodegen.apply(
            gtcpp, gt_backend_t=self.backend.GT_BACKEND_T
        )
        bindings = GTCppBindingsCodegen.apply(
            gtcpp, module_name=self.module_name, backend=self.backend
        )
        bindings_ext = ".cu" if self.backend.GT_BACKEND_T == "gpu" else ".cpp"
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings" + bindings_ext: bindings},
        }


class GTCppBindingsCodegen(codegen.TemplatedGenerator):
    def __init__(self):
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    def visit_DataType(self, dtype: DataType, **kwargs):
        return gtcpp_codegen.GTCppCodegen().visit_DataType(dtype)

    def visit_FieldDecl(self, node: gtcpp.FieldDecl, **kwargs):
        backend = kwargs["backend"]
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
                    backend=backend,
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

    Program = bindings_main_template()

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
