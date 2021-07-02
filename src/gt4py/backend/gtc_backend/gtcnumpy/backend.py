# -*- coding: utf-8 -*-
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type, Union, cast

from eve.codegen import format_source
from gt4py.backend.base import BaseBackend, BaseModuleGenerator, CLIBackendMixin, register
from gt4py.backend.debug_backend import (
    debug_is_compatible_layout,
    debug_is_compatible_type,
    debug_layout,
)
from gtc.gtir_to_oir import GTIRToOIR
from gtc.passes.gtir_legacy_extents import compute_legacy_extents
from gtc.passes.oir_pipeline import OirPipeline
from gtc.python import npir
from gtc.python.npir_gen import NpirGen
from gtc.python.oir_to_npir import OirToNpir


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCModuleGenerator(BaseModuleGenerator):
    def generate_imports(self) -> str:
        comp_pkg = (
            self.builder.caching.module_prefix + "computation" + self.builder.caching.module_postfix
        )
        return "\n".join(
            [
                *super().generate_imports().splitlines(),
                "import sys",
                "import pathlib",
                "import numpy",
                "path_backup = sys.path.copy()",
                "sys.path.append(str(pathlib.Path(__file__).parent))",
                f"import {comp_pkg} as computation",
                "print(repr(computation))",
                "sys.path = path_backup",
                "del path_backup",
            ]
        )

    def generate_implementation(self) -> str:
        params = [f"{p.name}={p.name}" for p in self.builder.gtir.params]
        params.extend(["_domain_=_domain_", "_origin_=_origin_"])
        return f"computation.run({', '.join(params)})"

    @property
    def backend(self) -> "GTCNumpyBackend":
        return cast(GTCNumpyBackend, self.builder.backend)


def recursive_write(root_path: pathlib.Path, tree: Dict[str, Union[str, dict]]):
    root_path.mkdir(parents=True, exist_ok=True)
    for key, value in tree.items():
        if isinstance(value, dict):
            recursive_write(root_path / key, value)
        else:
            src_path = root_path / key
            src_path.write_text(cast(str, value))


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
    USE_LEGACY_TOOLCHAIN = False
    GTIR_KEY = "gtc:gtir"

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        computation_name = (
            self.builder.caching.module_prefix
            + "computation"
            + self.builder.caching.module_postfix
            + ".py"
        )
        return {
            computation_name: format_source(
                "python",
                NpirGen.apply(self.npir, field_extents=compute_legacy_extents(self.builder.gtir)),
            ),
        }

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        super().generate_bindings(language_name)
        return {self.builder.module_path.name: self.make_module_source()}

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)
        src_dir = self.builder.module_path.parent
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            src_dir.mkdir(parents=True, exist_ok=True)
            recursive_write(src_dir, self.generate_computation())
        return self.make_module()

    def _make_npir(self) -> npir.Computation:
        return OirToNpir().visit(
            # TODO (ricoh) apply optimizations, skip only the ones that fail
            OirPipeline(GTIRToOIR().visit(self.builder.gtir)).apply([])
        )

    @property
    def npir(self) -> npir.Computation:
        key = "gtcnumpy:npir"
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_npir()})
        return self.builder.backend_data[key]
