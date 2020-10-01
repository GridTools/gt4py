from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Type, Union

from gt4py.backend.base import BaseBackend, CLIBackendMixin, register
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
class GTCPythonBackend(BaseBackend, CLIBackendMixin):
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
        source = self.make_module_source()
        return {filename: source}

    def make_module_source(
        self, *, args_data: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        return GTCPyModuleGenerator(self.builder)(MOCK_ARG_INFO)

    def generate(self) -> Type["StencilObject"]:
        self.builder.module_path.parent.mkdir(parents=True, exist_ok=True)
        self.builder.module_path.write_text(self.make_module_source())
        return self._load()
