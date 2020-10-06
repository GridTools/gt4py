from gt4py.backend.debug_backend import DebugModuleGenerator
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.backend.gtc_backend.python_naive_codegen import PythonNaiveCodegen
from devtools import debug  # TODO remove


class GTCPyModuleGenerator(DebugModuleGenerator):
    """Generate a python stencil module loadable by gt4py."""

    def generate_implementation(self) -> str:
        debug(self.builder.definition_ir)
        debug(DefIRToGTIR.apply(self.builder.definition_ir))
        implementation = PythonNaiveCodegen.apply(DefIRToGTIR.apply(self.builder.definition_ir))
        print(implementation)
        return implementation
