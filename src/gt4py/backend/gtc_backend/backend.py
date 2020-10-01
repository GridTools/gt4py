from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type, Union

from gt4py.backend.base import CLIBackendMixin, register
from gt4py.backend.debug_backend import (
    debug_is_compatible_layout,
    debug_is_compatible_type,
    debug_layout,
)

from .mock_arg_info import MOCK_ARG_INFO
from .py_module_generator import GTCPyModuleGenerator


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


@register
class GTCPythonBackend(CLIBackendMixin):
    """Pure python backend using gtc."""

    name = "gtc:py"
    options: ClassVar[Dict[str, Any]] = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": debug_layout,
        "is_compatible_layout": debug_is_compatible_layout,
        "is_compatible_type": debug_is_compatible_type,
    }
    languages = {"computation": "python", "bindings": []}

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        return super().generate_bindings(language_name)

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        filename = self.builder.module_path.name
        source = GTCPyModuleGenerator(self.builder)(MOCK_ARG_INFO)
        return {filename: source}

    def generate(self) -> Type["StencilObject"]:
        pass

    def load(self) -> Type["StencilObject"]:
        pass
