from gt4py.backend.debug_backend import DebugModuleGenerator
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.backend.gtc_backend.gtir_to_gtcpp import GTIRToGTCpp
from gt4py.backend.gtc_backend.gtcpp_codegen import GTCppCodegen
from gt4py.backend.gtc_backend.python_naive_codegen import PythonNaiveCodegen
from devtools import debug  # TODO remove
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from gt4py.backend import BaseModuleGenerator
from gt4py.utils.text import TextBlock, format_source

from .stencil_object_snippet_generators import (
    ACCESSOR_CLASS_SRC,
    ComputationCallGenerator,
    DomainInfoGenerator,
    FieldInfoGenerator,
    ParameterInfoGenerator,
    RunBodyGenerator,
)


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


class GTCPyModuleGenerator(BaseModuleGenerator):
    """Generate a python stencil module loadable by gt4py."""

    def generate_implementation(self) -> str:
        debug(self.builder.definition_ir)
        defir = DefIRToGTIR.apply(self.builder.definition_ir)
        gtcpp = GTIRToGTCpp().visit(defir)
        debug(gtcpp)
        print(GTCppCodegen.apply(gtcpp))
        implementation = PythonNaiveCodegen.apply(defir)
        print(implementation)
        return implementation

        # TODO Split: above code is the gtcpp one, below is python backend
        # source = TextBlock(indent_size=self.TEMPLATE_INDENT_SIZE)
        # source.append(RunBodyGenerator().apply(self.backend.gtc_ir))
        # source.empty_line(steps=2)
        # source.append(ComputationCallGenerator.apply(self.backend.gtc_ir))
        # return source.text

    def generate_imports(self) -> str:
        imports = super().generate_imports().split("\n")
        imports.extend(
            [
                "import sys",
                f"sys.path.append('{self.builder.module_path.parent}')",
                "import computation",
            ]
        )
        return "\n".join(imports)

    def generate_domain_info(self) -> str:
        return DomainInfoGenerator().apply(self.backend.gtc_ir)

    def generate_field_info(self) -> str:
        return FieldInfoGenerator().apply(self.backend.gtc_ir)

    def generate_parameter_info(self) -> str:
        return ParameterInfoGenerator().apply(self.backend.gtc_ir)

    def _get_options(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.builder.options.as_dict().items()
            if key not in ["build_info"]
        }

    # following type ignore is due to intentionally incompatible method signature
    def __call__(self, builder: Optional["StencilBuilder"] = None) -> str:  # type: ignore
        if builder:
            self._builder = builder

        return format_source(
            self.template.render(
                imports=self.generate_imports(),
                module_members=self.generate_module_members(),
                class_name=self.builder.class_name,
                class_members=self.generate_class_members(),
                docstring=self.builder.definition_ir.docstring,
                gt_backend=self.backend.name,
                gt_source=repr(str(self.generate_implementation())),
                gt_domain_info=self.generate_domain_info(),
                gt_field_info=self.generate_field_info(),
                gt_parameter_info=self.generate_parameter_info(),
                gt_constants=repr({}),
                gt_options=self._get_options(),
                stencil_signature=self.generate_signature(),
                field_names=self.backend.gtc_ir.param_names,
                param_names=[],
                pre_run=self.generate_pre_run(),
                post_run=self.generate_post_run(),
                implementation=self.generate_implementation(),
            ),
            line_length=self.SOURCE_LINE_LENGTH,
        )

    def generate_module_members(self) -> str:
        return ACCESSOR_CLASS_SRC

    @property
    def backend(self):
        from .backend import GTCPythonBackend

        return cast(GTCPythonBackend, self.builder.backend)
