from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type, Union, Tuple, cast, Optional

from gt4py.backend.base import BaseBackend, CLIBackendMixin, register
from gt4py.backend.debug_backend import (
    debug_is_compatible_layout,
    debug_is_compatible_type,
    debug_layout,
)

from gt4py.backend.gt_backends import (
    make_x86_layout_map,
    x86_is_compatible_layout,
    gtcpu_is_compatible_type,
    BaseGTBackend,
)

from .defir_to_gtir import DefIRToGTIR
from .fields_metadata_pass import FieldsMetadataPass
from .gtir import Computation
from .py_module_generator import GTCPyModuleGenerator
from .python_naive_codegen import PythonNaiveCodegen

from gt4py.backend import pyext_builder

from gt4py import utils as gt_utils
from gt4py import gt2_src_manager

from gt4py.backend.gtc_backend import gtc_gt_ext_generator

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


@register
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

    PYEXT_GENERATOR_CLASS = gtc_gt_ext_generator.GTCGTExtGenerator

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return self.make_extension(uses_cuda=False)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        implementation_ir = self.builder.implementation_ir

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt2_src_manager.has_gt_sources() and not gt2_src_manager.install_gt_sources():
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]
        if implementation_ir.has_effect:
            pyext_module_name, pyext_file_path = self.generate_extension()
        else:
            # if computation has no effect, there is no need to create an extension
            pyext_module_name, pyext_file_path = None, None

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )

    def make_extension(self, *, uses_cuda: bool = False) -> Tuple[str, str]:
        # Generate source
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_sources: Dict[str, Any] = self.make_extension_sources()
            gt_pyext_sources = {**gt_pyext_sources["computation"], **gt_pyext_sources["bindings"]}
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_sources = {
                key: gt_utils.NOTHING for key in self.PYEXT_GENERATOR_CLASS.TEMPLATE_FILES.keys()
            }

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
            ),
        )

        result = self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)
        return result

    def make_extension_sources(self) -> Dict[str, Dict[str, str]]:
        """Generate the source for the stencil independently from use case."""
        if "computation_src" in self.builder.backend_data:
            return self.builder.backend_data["computation_src"]
        class_name = (
            self.pyext_class_name if self.builder.stencil_id else self.builder.options.name
        )
        module_name = (
            self.pyext_module_name
            if self.builder.stencil_id
            else f"{self.builder.options.name}_pyext"
        )
        gt_pyext_generator = self.PYEXT_GENERATOR_CLASS(
            class_name, module_name, self.GT_BACKEND_T, self.builder.options
        )
        gt_pyext_sources = gt_pyext_generator(self.gtc_ir)
        final_ext = ".cu" if self.languages and self.languages["computation"] == "cuda" else ".cpp"
        comp_src = gt_pyext_sources["computation"]
        for key in [k for k in comp_src.keys() if k.endswith(".src")]:
            comp_src[key.replace(".src", final_ext)] = comp_src.pop(key)
        self.builder.backend_data["computation_src"] = gt_pyext_sources
        return gt_pyext_sources

    @property
    def gtc_ir(self) -> Computation:
        if "gtc_ir" in self.builder.backend_data:
            return self.builder.backend_data["gtc_ir"]
        return self.builder.with_backend_data(
            {"gtc_ir": FieldsMetadataPass().visit(DefIRToGTIR.apply(self.builder.definition_ir))}
        ).backend_data["gtc_ir"]
