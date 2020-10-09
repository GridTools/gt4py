from typing import TYPE_CHECKING, Any, Dict, Optional

from gt4py.backend import BaseModuleGenerator
from gt4py.utils.text import TextBlock, format_source

from .defir_to_gtir import DefIRToGTIR
from .stencil_object_snippet_generator import StencilObjectSnippetGenerator


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


class GTCPyModuleGenerator(BaseModuleGenerator):
    """Generate a python stencil module loadable by gt4py."""

    def generate_implementation(self) -> str:
        source = TextBlock(indent_size=self.TEMPLATE_INDENT_SIZE)
        source.append(f"computation.run({self.builder.backend.gtc_ir.signature})")
        return source

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

    def _get_options(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.builder.options.as_dict().items()
            if key not in ["build_info"]
        }

    def __call__(self, builder: Optional["StencilBuilder"] = None) -> str:
        if builder:
            self._builder = builder

        gtc_ir = self.builder.backend.gtc_ir
        gen = StencilObjectSnippetGenerator()

        return format_source(
            self.template.render(
                imports=self.generate_imports(),
                module_members=self.generate_module_members(),
                class_name=self.builder.class_name,
                class_members=self.generate_class_members(),
                docstring=self.builder.definition_ir.docstring,
                gt_backend=self.builder.backend.name,
                gt_source=repr(str(self.generate_implementation())),
                gt_domain_info=gtc_ir.domain_info,
                gt_field_info=gen.apply(gtc_ir.fields_metadata),
                gt_parameter_info=repr(gtc_ir.parameter_info),
                gt_constants=repr(gtc_ir.constants),
                gt_options=self._get_options(),
                stencil_signature=self.generate_signature(),
                field_names=gtc_ir.field_names,
                param_names=gtc_ir.param_names,
                pre_run=self.generate_pre_run(),
                post_run=self.generate_post_run(),
                implementation=self.generate_implementation(),
            ),
            line_length=self.SOURCE_LINE_LENGTH,
        )
