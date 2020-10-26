from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type, Union, cast

from gt4py.backend.base import BaseBackend, CLIBackendMixin, register
from gt4py.backend.debug_backend import (
    debug_is_compatible_layout,
    debug_is_compatible_type,
    debug_layout,
)

from .defir_to_gtir import DefIRToGTIR
from .fields_metadata_pass import FieldsMetadataPass
from .gtir import Computation
from .py_module_generator import GTCPyModuleGenerator
from .python_naive_codegen import PythonNaiveCodegen


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


@register
class GTCPythonBackend(BaseBackend, CLIBackendMixin):
    """Pure python backend using gtc."""

    name = "gtc:py"
    # Intentionally does not define a MODULE_GENERATOR_CLASS
    options: ClassVar[Dict[str, Any]] = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": debug_layout,
        "is_compatible_layout": debug_is_compatible_layout,
        "is_compatible_type": debug_is_compatible_type,
    }
    languages = {"computation": "python", "bindings": ["python"]}

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        if language_name == "python":
            return {self.builder.module_path.name: GTCPyModuleGenerator(builder=self.builder)()}
        return super().generate_bindings(language_name)

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        filename = "computation.py"
        source = PythonNaiveCodegen().apply(self.gtc_ir)
        return {filename: source}

    def generate(self) -> Type["StencilObject"]:
        self.builder.module_path.parent.mkdir(parents=True, exist_ok=True)
        computation_filename, computation_source = list(self.generate_computation().items())[0]
        self.builder.module_path.parent.joinpath(computation_filename).write_text(
            cast(str, computation_source)
        )
        bindings_filename, bindings_source = list(
            self.generate_bindings(language_name="python").items()
        )[0]
        self.builder.module_path.write_text(cast(str, bindings_source))
        return self._load()

    @property
    def gtc_ir(self) -> Computation:
        if "gtc_ir" in self.builder.backend_data:
            return self.builder.backend_data["gtc_ir"]
        return self.builder.with_backend_data(
            {"gtc_ir": FieldsMetadataPass().visit(DefIRToGTIR.apply(self.builder.definition_ir))}
        ).backend_data["gtc_ir"]
