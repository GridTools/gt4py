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
from eve.codegen import JinjaTemplate
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
from gt4py.backend.gtc_backend.common import bindings_main_template, pybuffer_to_sid
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.ir.nodes import StencilDefinition
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.gtcpp import gtcpp, gtcpp_codegen, oir_to_gtcpp
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
from gtc.passes.oir_pipeline import DefaultPipeline


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


def make_gtcpp_ir(definition_ir: StencilDefinition, backend) -> gtcpp.Program:
    gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
    base_oir = gtir_to_oir.GTIRToOIR().visit(gtir)
    oir_pipeline = backend.builder.options.backend_opts.get(
        "oir_pipeline", DefaultPipeline(skip=[FillFlushToLocalKCaches])
    )
    oir = oir_pipeline.run(base_oir)
    gtcpp = oir_to_gtcpp.OIRToGTCpp().visit(oir)
    return gtcpp


class GTCGTExtGenerator:
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtcpp = make_gtcpp_ir(definition_ir, self.backend)
        format_source = self.backend.builder.options.format_source
        implementation = gtcpp_codegen.GTCppCodegen.apply(
            gtcpp, gt_backend_t=self.backend.GT_BACKEND_T, format_source=format_source
        )
        bindings = GTCppBindingsCodegen.apply(
            gtcpp, module_name=self.module_name, backend=self.backend, format_source=format_source
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
        if kwargs.get("format_source", True):
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code


class FortranBindingsCodegen(codegen.TemplatedGenerator):
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
            dtype = self.visit(node.dtype)
            stride_kind = self.unique_index()
            if kwargs["external_arg"]:
                return "gridtools::fortran_array_view<{dtype}, {sid_ndim}, gridtools::integral_constant<int, {stride_kind}>> {name}".format(
                    name=node.name, sid_ndim=sid_ndim, dtype=dtype, stride_kind=stride_kind
                )
            else:
                return node.name

    def visit_GlobalParamDecl(self, node: gtcpp.GlobalParamDecl, **kwargs):
        if "external_arg" in kwargs:
            if kwargs["external_arg"]:
                return "{dtype} {name}".format(name=node.name, dtype=self.visit(node.dtype))
            else:
                return "gridtools::stencil::make_global_parameter({name})".format(name=node.name)

    def visit_Program(self, node: gtcpp.Program, **kwargs):
        # assert "module_name" in kwargs
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
        #include "computation.hpp"
        #include <cpp_bindgen/export.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/storage/adapter/fortran_array_view.hpp>

        namespace {
        void ${name}_impl(unsigned int nx, unsigned int ny, unsigned int nz,
                    ${','.join(entry_params)}) {
        auto computation = ${name}({nx, ny, nz});
        computation(${','.join(sid_params)});
        }
        BINDGEN_EXPORT_BINDING_WRAPPED(${3+len(entry_params)}, ${name}, ${name}_impl);
        } // namespace
        """
    )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = cls().visit(root, **kwargs)
        if kwargs.get("format_source", True):
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code


def make_cmake_lists(libname, filename):
    return as_mako(
        """
cmake_minimum_required(VERSION 3.16.2)

project(sample LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  gridtools
  GIT_REPOSITORY https://github.com/GridTools/gridtools.git
  GIT_TAG        master # consider replacing master by a tagged version
)
FetchContent_MakeAvailable(gridtools)

FetchContent_Declare(
  cpp_bindgen
  GIT_REPOSITORY https://github.com/GridTools/cpp_bindgen.git
  GIT_TAG        master # consider replacing master by a tagged version
)
FetchContent_MakeAvailable(cpp_bindgen)

bindgen_add_library(${libname}_module SOURCES ${filename})
target_link_libraries(${libname}_module PRIVATE GridTools::stencil_cpu_kfirst)

install_cpp_bindgen_targets(
  EXPORT ${libname}_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

install(
  TARGETS ${libname}_module
  EXPORT ${libname}_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

    """
    ).render_values(libname=libname, filename=filename)


class GTCGTBaseBackend(BaseGTBackend):  # , CLIBackendMixin):
    options = BaseGTBackend.GT_BACKEND_OPTS
    PYEXT_GENERATOR_CLASS = GTCGTExtGenerator  # type: ignore
    USE_LEGACY_TOOLCHAIN = False

    def _generate_extension(self, uses_cuda: bool) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=uses_cuda)

    def generate_bindings(self, language_name: str, ir):
        if language_name != "fortran":
            return super().generate_bindings(language_name, ir)
        print("Generating Fortran bindings")
        dir_name = f"{self.builder.options.name}_src"
        return {
            dir_name: {
                "bindings.cpp": FortranBindingsCodegen.apply(make_gtcpp_ir(ir, self), backend=self),
                "CMakeLists.txt": make_cmake_lists(ir.name, "bindings.cpp"),
            }
        }

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
    options = {**BaseGTBackend.GT_BACKEND_OPTS, "device_sync": {"versioning": True, "type": bool}}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": make_cuda_layout_map,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=True)
