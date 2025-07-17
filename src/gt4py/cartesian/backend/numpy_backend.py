# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Type, Union, cast

from gt4py import storage as gt_storage
from gt4py.cartesian.backend.base import BaseBackend, CLIBackendMixin, register
from gt4py.cartesian.backend.module_generator import BaseModuleGenerator
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.cartesian.gtc.numpy import npir
from gt4py.cartesian.gtc.numpy.npir_codegen import NpirCodegen
from gt4py.cartesian.gtc.numpy.oir_to_npir import OirToNpir
from gt4py.cartesian.gtc.numpy.scalars_to_temps import ScalarsToTemporaries
from gt4py.cartesian.gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gt4py.cartesian.gtc.passes.oir_pipeline import DefaultPipeline, OirPipeline
from gt4py.eve.codegen import format_source


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_object import StencilObject


class ModuleGenerator(BaseModuleGenerator):
    def generate_imports(self) -> str:
        comp_pkg = (
            self.builder.caching.module_prefix + "computation" + self.builder.caching.module_postfix
        )
        return "\n".join(
            [
                *super().generate_imports().splitlines(),
                "import pathlib",
                "from gt4py.cartesian.utils import make_module_from_file",
                f'computation = make_module_from_file("{comp_pkg}", pathlib.Path(__file__).parent / "{comp_pkg}.py")',
            ]
        )

    def generate_implementation(self) -> str:
        params = [f"{p.name}={p.name}" for p in self.builder.gtir.params]
        params.extend(["_domain_=_domain_", "_origin_=_origin_"])
        return f"computation.run({', '.join(params)})"

    @property
    def backend(self) -> NumpyBackend:
        return cast(NumpyBackend, self.builder.backend)


def recursive_write(root_path: pathlib.Path, tree: Dict[str, Union[str, dict]]):
    root_path.mkdir(parents=True, exist_ok=True)
    for key, value in tree.items():
        if isinstance(value, dict):
            recursive_write(root_path / key, value)
        else:
            src_path = root_path / key
            src_path.write_text(value)


@register
class NumpyBackend(BaseBackend, CLIBackendMixin):
    """NumPy backend using gtc."""

    name = "numpy"
    options: ClassVar[Dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
        # TODO: Implement this option in source code
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
    storage_info = gt_storage.layout.NaiveCPULayout
    languages: ClassVar[dict] = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = ModuleGenerator

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        computation_name = (
            self.builder.caching.module_prefix
            + "computation"
            + self.builder.caching.module_postfix
            + ".py"
        )

        ignore_np_errstate = self.builder.options.backend_opts.get("ignore_np_errstate", True)
        source = NpirCodegen.apply(self.npir, ignore_np_errstate=ignore_np_errstate)
        if self.builder.options.format_source:
            source = format_source("python", source)

        return {computation_name: source}

    def generate_bindings(self, language_name: str) -> Dict[str, Union[str, Dict]]:
        super().generate_bindings(language_name)
        return {self.builder.module_path.name: self.make_module_source()}

    def generate(self) -> Type[StencilObject]:
        self.check_options(self.builder.options)
        src_dir = self.builder.module_path.parent
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            src_dir.mkdir(parents=True, exist_ok=True)
            recursive_write(src_dir, self.generate_computation())
        return self.make_module()

    def _make_npir(self) -> npir.Computation:
        base_oir = GTIRToOIR().visit(self.builder.gtir)
        oir_pipeline = self.builder.options.backend_opts.get(
            "oir_pipeline",
            DefaultPipeline(
                skip=[IJCacheDetection, KCacheDetection, PruneKCacheFills, PruneKCacheFlushes]
            ),
        )
        oir_node = oir_pipeline.run(base_oir)
        base_npir = OirToNpir().visit(oir_node)
        npir_node = ScalarsToTemporaries().visit(base_npir)
        return npir_node

    @property
    def npir(self) -> npir.Computation:
        key = "gtcnumpy:npir"
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_npir()})
        return self.builder.backend_data[key]
