import numbers
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union, cast

from gt4py.backend.base import BaseBackend, BaseModuleGenerator, CLIBackendMixin, register
from gt4py.backend.debug_backend import (
    debug_is_compatible_layout,
    debug_is_compatible_type,
    debug_layout,
)
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.backend.gtc_backend.stencil_object_snippet_generators import (
    DomainInfoGenerator,
    FieldInfoGenerator,
    ParameterInfoGenerator,
)
from gt4py.gtc import gtir
from gt4py.gtc.gtir_to_oir import GTIRToOIR
from gt4py.gtc.passes.fields_metadata_pass import FieldsMetadataPass
from gt4py.gtc.passes.gtir_dtype_resolver import resolve_dtype
from gt4py.gtc.passes.gtir_upcaster import upcast
from gt4py.gtc.python import npir
from gt4py.gtc.python.npir_gen import NpirGen
from gt4py.gtc.python.oir_to_npir import OirToNpir
from gt4py.utils import text


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder
    from gt4py.stencil_object import StencilObject


class GTCModuleGenerator(BaseModuleGenerator):
    # type ignore reason: signature differs from super().__call__ on purpose.
    def __call__(self, builder: Optional["StencilBuilder"] = None, **kwargs: Any) -> str:  # type: ignore # noqa
        return text.format_source(
            self.template.render(
                imports=self.generate_imports(),
                module_members=self.generate_module_members(),
                class_name=self.generate_class_name(),
                docstring=self.generate_docstring(),
                gt_backend=self.generate_gt_backend(),
                gt_source=self.generate_gt_source(),
                gt_domain_info=self.generate_gt_domain_info(),
                gt_parameter_info=self.generate_gt_parameter_info(),
                gt_constants=self.generate_gt_constants(),
                gt_options=self.generate_gt_options(),
                gt_field_info=self.generate_gt_field_info(),
                stencil_signature=self.generate_signature(),
                field_names=self.generate_field_names(),
                param_names=[],
                pre_run=self.generate_pre_run(),
                post_run=self.generate_post_run(),
                implementation=self.generate_implementation(),
            ),
            line_length=self.SOURCE_LINE_LENGTH,
        )

    def generate_imports(self) -> str:
        return "\n".join(
            [
                *super().generate_imports().splitlines(),
                "import sys",
                "import pathlib",
                "sys.path.append(str(pathlib.Path(__file__).parent))",
                "import numpy",
                "import computation",
            ]
        )

    def generate_module_members(self) -> str:
        return super().generate_module_members()

    def generate_class_name(self) -> str:
        return self.builder.class_name

    def generate_docstring(self) -> str:
        return self.builder.definition_ir.docstring

    def generate_gt_backend(self) -> str:
        return self.backend_name

    def generate_gt_source(self) -> Dict[str, str]:
        if self.builder.definition_ir.sources is None:
            return {}
        return {
            key: text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
            for key, value in self.builder.definition_ir.sources
        }

    def generate_gt_domain_info(self) -> str:
        return DomainInfoGenerator.apply(self.backend.gtir)

    def generate_gt_field_info(self) -> str:
        # infos = [
        #    f"'{name}': FieldInfo(access=AccessKind.{READ_WRITE}, boundary=
        # ]
        # return FieldInfoGenerator.apply(self.backend.gtir)
        return self.args_data["field_info"]

    def generate_gt_parameter_info(self) -> str:
        return ParameterInfoGenerator.apply(self.backend.gtir)

    def generate_gt_constants(self) -> Dict[str, str]:
        if not self.builder.definition_ir.externals:
            return {}
        return {
            name: repr(value)
            for name, value in self.builder.definition_ir.externals.items()
            if isinstance(value, numbers.Number)
        }

    def generate_gt_options(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.builder.options.as_dict().items()
            if key not in ["build_info"]
        }

    def generate_field_names(self) -> List[str]:
        return self.backend.npir.field_params

    def generate_param_names(self) -> List[str]:
        return [param.name for param in self.backend.gtir.params]

    def generate_implementation(self) -> str:
        params = [f"{p.name}={p.name}" for p in self.backend.gtir.params]
        params.extend(["_domain_=_domain_", "_origin_=_origin_"])
        return f"computation.run({', '.join(params)})"

    @property
    def backend(self) -> "GTCNumpyBackend":
        return cast(GTCNumpyBackend, self.builder.backend)


@register
class GTCNumpyBackend(BaseBackend, CLIBackendMixin):
    """NumPy backend using gtc."""

    name = "gtc:numpy"
    options: ClassVar[Dict[str, Any]] = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": debug_layout,
        "is_compatible_layout": debug_is_compatible_layout,
        "is_compatible_type": debug_is_compatible_type,
    }
    languages = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = GTCModuleGenerator
    GTIR_KEY = "gtc:gtir"

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        computation_name = "computation.py"
        return {computation_name: NpirGen.apply(self.npir)}

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        super().generate_bindings(language_name)
        return {self.builder.module_path.name: self.make_module_source()}

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)
        src_dir = self.builder.module_path.parent
        computation_src = list(self.generate_computation().items())
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            src_dir.mkdir(parents=True, exist_ok=True)
            for filename, src in computation_src:
                src_path = src_dir / filename
                src_path.write_text(src)
        return self.make_module()

    # type ignore reason: signature differs from super on purpose
    def make_module_source(self) -> str:  # type: ignore
        args_data = self.make_args_data_from_iir(self.builder.implementation_ir)
        return self.MODULE_GENERATOR_CLASS(self.builder)(args_data)

    def _make_gtir(self) -> gtir.Stencil:
        gtir = FieldsMetadataPass().visit(DefIRToGTIR.apply(self.builder.definition_ir))
        return upcast(resolve_dtype(gtir))

    def _make_npir(self) -> npir.Computation:
        oir = GTIRToOIR().visit(self.gtir)
        return OirToNpir().visit(oir)

    @property
    def gtir(self) -> gtir.Stencil:
        key = self.GTIR_KEY
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_gtir()})
        return self.builder.backend_data[key]

    @property
    def npir(self) -> npir.Computation:
        key = "gtcnumpy:npir"
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_npir()})
        return self.builder.backend_data[key]
