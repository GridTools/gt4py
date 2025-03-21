# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from typing import TYPE_CHECKING, Any, ClassVar, Type, Union

from gt4py import storage
from gt4py.cartesian.backend.base import BaseBackend, CLIBackendMixin, register
from gt4py.cartesian.backend.numpy_backend import ModuleGenerator
from gt4py.cartesian.gtc.debug.debug_codegen import DebugCodeGen
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.cartesian.gtc.passes.oir_optimizations.horizontal_execution_merging import (
    HorizontalExecutionMerging,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.temporaries import LocalTemporariesToScalars
from gt4py.cartesian.gtc.passes.oir_pipeline import OirPipeline
from gt4py.eve.codegen import format_source


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_object import StencilObject


@register
class DebugBackend(BaseBackend, CLIBackendMixin):
    """Debug backend using plain python loops."""

    name = "debug"
    options: ClassVar[dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
    storage_info = storage.layout.NaiveCPULayout
    languages = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = ModuleGenerator

    def generate_computation(self) -> dict[str, Union[str, dict]]:
        computation_name = (
            f"{self.builder.caching.module_prefix}"
            + f"computation{self.builder.caching.module_postfix}.py"
        )

        oir = GTIRToOIR().visit(self.builder.gtir)
        oir = HorizontalExecutionMerging().visit(oir)
        oir = LocalTemporariesToScalars().visit(oir)
        source_code = DebugCodeGen().visit(oir)

        if self.builder.options.format_source:
            source_code = format_source("python", source_code)

        return {computation_name: source_code}

    def generate_bindings(self, language_name: str) -> dict[str, Union[str, dict]]:
        super().generate_bindings(language_name)
        return {self.builder.module_path.name: self.make_module_source()}

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)
        src_dir = self.builder.module_path.parent
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            src_dir.mkdir(parents=True, exist_ok=True)
            self.recursive_write(src_dir, self.generate_computation())
        return self.make_module()
