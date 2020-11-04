from . import gtcppir
from eve import codegen
from eve.codegen import MakoTemplate as as_mako


class GTCppBindingsCodegen(codegen.TemplatedGenerator):
    # ParamArg = as_fmt("py::buffer {name}, std::array<gt::unit_t,3> {name}_origin")

    def visit_Computation(self, node: gtcppir.Computation, **kwargs):
        assert "module_name" in kwargs
        entry_params = [
            "py::buffer {name}, std::array<gt::uint_t,3> {name}_origin".format(name=p.name)
            for p in node.parameters
        ]
        sid_params = ["gt::as_sid<double, 3>({name})".format(name=p.name) for p in node.parameters]
        return self.generic_visit(
            node,
            entry_params=entry_params,
            sid_params=sid_params,
            **kwargs,
        )

    Computation = as_mako(
        """#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gridtools/storage/adapter/python_sid_adapter.hpp>
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
