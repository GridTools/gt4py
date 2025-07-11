# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from gt4py.cartesian import backend
from gt4py.cartesian.backend import python_common as py_common
from gt4py.cartesian.gtc import passes
from gt4py.cartesian.gtc.debug.debug_codegen import DebugCodeGen
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.cartesian.gtc.passes import oir_optimizations
from gt4py.eve import codegen
from gt4py.storage import layout


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_object import StencilObject


@backend.register
class DebugBackend(backend.BaseBackend):
    """Debug backend using plain python loops."""

    name = "debug"
    options: ClassVar[dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": passes.OirPipeline},
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
    storage_info = layout.NaiveCPULayout
    languages: ClassVar[dict[str, Any]] = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = py_common.PythonModuleGenerator

    def _generate_computation(self) -> dict[str, str | dict]:
        oir = GTIRToOIR().visit(self.builder.gtir)
        oir = oir_optimizations.HorizontalExecutionMerging().visit(oir)
        oir = oir_optimizations.LocalTemporariesToScalars().visit(oir)
        source_code = DebugCodeGen().visit(oir)

        if self.builder.options.format_source:
            source_code = codegen.format_source("python", source_code)

        caching = self.builder.caching
        computation_name = f"{caching.module_prefix}computation{caching.module_postfix}.py"
        return {computation_name: source_code}

    def generate(self) -> type[StencilObject]:
        self.check_options(self.builder.options)
        src_dir = self.builder.module_path.parent

        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            py_common.recursive_write(src_dir, self._generate_computation())

        return self.make_module()
