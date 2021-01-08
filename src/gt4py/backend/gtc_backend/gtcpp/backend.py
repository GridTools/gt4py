from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple, Type

from eve import codegen
from eve.codegen import MakoTemplate as as_mako

from gt4py import backend as gt_backend
from gt4py import gt2_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin
from gt4py.backend.gt_backends import (
    gtcpu_is_compatible_type,
    make_x86_layout_map,
    x86_is_compatible_layout,
)
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.gtc import gtir_to_oir, passes
from gt4py.gtc.common import DataType
from gt4py.gtc.gtcpp import gtcpp, gtcpp_codegen, oir_to_gtcpp
from gt4py.gtc.passes.gtir_dtype_resolver import resolve_dtype
from gt4py.gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gt4py.gtc.passes.gtir_upcaster import upcast


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCGTExtGenerator:
    COMPUTATION_FILES = ["computation.hpp"]
    BINDINGS_FILES = ["bindings.cpp"]

    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

    def __call__(self, definition_ir) -> Dict[str, Dict[str, str]]:
        gtir = passes.FieldsMetadataPass().visit(DefIRToGTIR.apply(definition_ir))
        gtir_without_unused_params = prune_unused_parameters(gtir)
        dtype_deduced = resolve_dtype(gtir_without_unused_params)
        upcasted = upcast(dtype_deduced)
        oir = gtir_to_oir.GTIRToOIR().visit(upcasted)
        gtcpp = oir_to_gtcpp.OIRToGTCpp().visit(oir)
        implementation = gtcpp_codegen.GTCppCodegen.apply(gtcpp)
        bindings = GTCppBindingsCodegen.apply(gtcpp, self.module_name)
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cc": bindings},
        }


class GTCppBindingsCodegen(codegen.TemplatedGenerator):
    def __init__(self):
        self._unique_index = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

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
                return "gt::sid::shift_sid_origin(gt::as_sid<{dtype}, 3, std::integral_constant<int, {unique_index}>>({name}), {name}_origin)".format(
                    name=node.name, dtype=self.visit(node.dtype), unique_index=self.unique_index()
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
        %if len(entry_params) > 0:
        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", [](std::array<gt::uint_t, 3> domain,
            ${','.join(entry_params)},
            py::object exec_info){
                ${name}(domain)(${','.join(sid_params)});
            }, "Runs the given computation");}
        %endif
        """
    )

    @classmethod
    def apply(cls, root, module_name, **kwargs) -> str:
        generated_code = cls().visit(root, module_name=module_name, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


@gt_backend.register
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
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=False)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt2_src_manager.has_gt_sources() and not gt2_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]

        # TODO add bypass if computation has no effect
        pyext_module_name, pyext_file_path = self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )
