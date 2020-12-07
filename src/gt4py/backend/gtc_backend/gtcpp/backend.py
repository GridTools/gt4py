from typing import Any, ClassVar, Dict, Optional, Tuple
from gt4py.backend import BaseGTBackend, CLIBackendMixin, pyext_builder
from gt4py.backend.base import register
from gt4py.backend.gt_backends import (
    gtcpu_is_compatible_type,
    make_x86_layout_map,
    x86_is_compatible_layout,
)
from gt4py import gt2_src_manager

from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR

from gt4py import utils as gt_utils
from gt4py.gtc import gtir, gtir_to_oir, passes
from gt4py.gtc.common import DataType
from gt4py.gtc.gtcpp import oir_to_gtcpp, gtcpp, gtcpp_codegen

from eve import codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py.gtc.passes.gtir_set_dtype import GTIRSetDtype

from devtools import debug


class GTCGTExtGenerator:
    COMPUTATION_FILES = ["computation.hpp"]
    BINDINGS_FILES = ["bindings.cpp"]

    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

    # TODO here definition IR should be the input
    def __call__(self, gtir: gtir.Stencil) -> Dict[str, Dict[str, str]]:
        # debug(gtir)
        dtype_deduced = GTIRSetDtype().visit(gtir)
        oir = gtir_to_oir.GTIRToOIR().visit(dtype_deduced)
        # debug(oir)
        gtcpp = oir_to_gtcpp.OIRToGTCpp().visit(oir)
        implementation = gtcpp_codegen.GTCppCodegen.apply(gtcpp)
        bindings = GTCppBindingsCodegen.apply(gtcpp, self.module_name)
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cc": bindings},
        }


class GTCppBindingsCodegen(codegen.TemplatedGenerator):
    # ParamArg = as_fmt("py::buffer {name}, std::array<gt::unit_t,3> {name}_origin")

    def visit_DataType(self, dtype: DataType, **kwargs):
        if dtype == DataType.INT64:
            return "long long"
        elif dtype == DataType.FLOAT64:
            return "double"
        elif dtype == DataType.FLOAT32:
            return "float"
        elif dtype == DataType.BOOL:
            return "bool"
        else:
            assert False

    def visit_FieldDecl(self, node: gtcpp.FieldDecl, **kwargs):
        if "external_arg" in kwargs:
            if kwargs["external_arg"]:
                return "py::buffer {name}, std::array<gt::uint_t,3> {name}_origin".format(
                    name=node.name
                )
            else:
                return "gt::sid::shift_sid_origin(gt::as_sid<{dtype}, 3>({name}), {name}_origin)".format(
                    name=node.name, dtype=self.visit(node.dtype)
                )

    def visit_GlobalParamDecl(self, node: gtcpp.GlobalParamDecl, **kwargs):
        if "external_arg" in kwargs:
            if kwargs["external_arg"]:
                return "{dtype} {name}".format(name=node.name, dtype=self.visit(node.dtype))
            else:
                return "gridtools::stencil::make_global_parameter({name})".format(name=node.name)

    def visit_Program(self, node: gtcpp.Program, **kwargs):
        assert "module_name" in kwargs
        entry_params = self.visit(node.parameters, external_arg=True)
        sid_params = self.visit(node.parameters, external_arg=False)
        return self.generic_visit(
            node,
            entry_params=entry_params,
            sid_params=sid_params,
            **kwargs,
        )

    Program = as_mako(
        """
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/sid/sid_shift_origin.hpp>
        #include "computation.hpp"
        namespace gt = gridtools;
        namespace py = ::pybind11;
        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", [](std::array<gt::uint_t, 3> domain,
            ${','.join(entry_params)},  py::object exec_info){
                ${name}(domain)(${','.join(sid_params)});
            }, "Runs the given computation");}
        """
    )

    @classmethod
    def apply(cls, root, module_name, **kwargs) -> str:
        generated_code = cls().visit(root, module_name=module_name, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


@register
class GTCGTBackend(BaseGTBackend, CLIBackendMixin):
    """GridTools python backend using gtc."""

    name = "gtc:gt"

    GT_BACKEND_T = "x86"
    options: ClassVar[Dict[str, Any]] = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }
    languages = {"computation": "c++", "bindings": ["python"]}

    PYEXT_GENERATOR_CLASS = GTCGTExtGenerator

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False)

    def generate(self):  # -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        implementation_ir = self.builder.implementation_ir

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt2_src_manager.has_gt_sources() and not gt2_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]
        if implementation_ir.has_effect:
            pyext_module_name, pyext_file_path = self.generate_extension()
        else:
            # if computation has no effect, there is no need to create an extension
            pyext_module_name, pyext_file_path = None, None

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    def make_extension(self, *, uses_cuda: bool = False) -> Tuple[str, str]:
        # Generate source
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_sources: Dict[str, Any] = self.make_extension_sources()
            gt_pyext_sources = {**gt_pyext_sources["computation"], **gt_pyext_sources["bindings"]}
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_sources = {
                key: gt_utils.NOTHING for key in self.PYEXT_GENERATOR_CLASS.TEMPLATE_FILES.keys()
            }

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
                gt_version=2,
            ),
        )

        result = self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)
        return result

    def make_extension_sources(self) -> Dict[str, Dict[str, str]]:
        """Generate the source for the stencil independently from use case."""
        if "computation_src" in self.builder.backend_data:
            return self.builder.backend_data["computation_src"]
        class_name = (
            self.pyext_class_name if self.builder.stencil_id else self.builder.options.name
        )
        module_name = (
            self.pyext_module_name
            if self.builder.stencil_id
            else f"{self.builder.options.name}_pyext"
        )
        gt_pyext_generator = self.PYEXT_GENERATOR_CLASS(
            class_name, module_name, self.GT_BACKEND_T, self.builder.options
        )
        gt_pyext_sources = gt_pyext_generator(self.gtc_ir)
        final_ext = ".cu" if self.languages and self.languages["computation"] == "cuda" else ".cpp"
        comp_src = gt_pyext_sources["computation"]
        for key in [k for k in comp_src.keys() if k.endswith(".src")]:
            comp_src[key.replace(".src", final_ext)] = comp_src.pop(key)
        self.builder.backend_data["computation_src"] = gt_pyext_sources
        return gt_pyext_sources

    @property
    def gtc_ir(self) -> gtir.Stencil:
        if "gtc_ir" in self.builder.backend_data:
            return self.builder.backend_data["gtc_ir"]
        return self.builder.with_backend_data(
            {
                "gtc_ir": passes.FieldsMetadataPass().visit(
                    DefIRToGTIR.apply(self.builder.definition_ir)
                )
            }
        ).backend_data["gtc_ir"]
