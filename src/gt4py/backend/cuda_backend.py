# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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
from gt4py import gt_src_manager
from gt4py.backend.base import CLIBackendMixin, register
from gt4py.backend.gtc_common import BackendCodegen, bindings_main_template, pybuffer_to_sid
from gt4py.utils.layout import layout_checker_factory
from gtc import gtir
from gtc.common import DataType
from gtc.cuir import cuir, cuir_codegen, extent_analysis, kernel_fusion
from gtc.cuir.oir_to_cuir import OIRToCUIR
from gtc.gtir_to_oir import GTIRToOIR
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_pipeline import DefaultPipeline

from .gtc_common import BaseGTBackend, CUDAPyExtModuleGenerator, make_cuda_layout_map


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class CudaExtGenerator(BackendCodegen):
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, stencil_ir: gtir.Stencil) -> Dict[str, Dict[str, str]]:
        stencil_ir = GtirPipeline(stencil_ir, self.backend.builder.stencil_id).full()
        base_oir = GTIRToOIR().visit(stencil_ir)
        oir_pipeline = self.backend.builder.options.backend_opts.get(
            "oir_pipeline",
            DefaultPipeline(skip=[NoFieldAccessPruning], add_steps=[FillFlushToLocalKCaches]),
        )
        oir_node = oir_pipeline.run(base_oir)
        cuir_node = OIRToCUIR().visit(oir_node)
        cuir_node = kernel_fusion.FuseKernels().visit(cuir_node)
        cuir_node = extent_analysis.CacheExtents().visit(cuir_node)
        format_source = self.backend.builder.options.format_source
        implementation = cuir_codegen.CUIRCodegen.apply(cuir_node, format_source=format_source)
        bindings = CudaBindingsCodegen.apply_codegen(
            cuir_node,
            module_name=self.module_name,
            backend=self.backend,
            format_source=format_source,
        )
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cu": bindings},
        }


class CudaBindingsCodegen(codegen.TemplatedGenerator):
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
                return "py::object {name}, std::array<gt::int_t,{sid_ndim}> {name}_origin".format(
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
                return "gridtools::stencil::global_parameter({name})".format(name=node.name)

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
    def apply_codegen(cls, root, *, module_name="stencil", backend, **kwargs) -> str:
        generated_code = cls(backend).visit(root, module_name=module_name, **kwargs)
        if kwargs.get("format_source", True):
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code


@register
class CudaBackend(BaseGTBackend, CLIBackendMixin):
    """CUDA backend using gtc."""

    name = "cuda"
    options = {**BaseGTBackend.GT_BACKEND_OPTS, "device_sync": {"versioning": True, "type": bool}}
    languages = {"computation": "cuda", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": make_cuda_layout_map,
        "is_optimal_layout": layout_checker_factory(make_cuda_layout_map),
    }
    PYEXT_GENERATOR_CLASS = CudaExtGenerator  # type: ignore
    MODULE_GENERATOR_CLASS = CUDAPyExtModuleGenerator
    GT_BACKEND_T = "gpu"

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(stencil_ir=self.builder.gtir, uses_cuda=True)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
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
