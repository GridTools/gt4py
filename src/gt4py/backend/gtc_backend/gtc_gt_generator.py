from typing import Dict

from . import gtir

from gt4py.backend.gtc_backend.gtcpp_codegen import GTCppCodegen
from gt4py.backend.gtc_backend.gtcpp_bindingsgen import GTCppBindingsCodegen
from gt4py.backend.gtc_backend.gtir_to_gtcpp import GTIRToGTCpp


class GTCGTExtGenerator:
    COMPUTATION_FILES = ["computation.hpp"]
    BINDINGS_FILES = ["bindings.cpp"]

    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

    # TODO here definition IR should be the input
    def __call__(self, gtir: gtir.Computation) -> Dict[str, Dict[str, str]]:
        gtcpp = GTIRToGTCpp().visit(gtir)
        implementation = GTCppCodegen.apply(gtcpp)
        bindings = GTCppBindingsCodegen.apply(gtcpp, self.module_name)
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings.cc": bindings},
        }
