# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from typing import TYPE_CHECKING, Any, ClassVar, Type, Union

from gt4py import storage
from gt4py.cartesian.backend import base as backend_base, numpy_backend
from gt4py.cartesian.gtc import gtir_to_oir
from gt4py.cartesian.gtc.debug import debug_codegen
from gt4py.cartesian.gtc.passes import oir_pipeline
from gt4py.cartesian.gtc.passes.oir_optimizations import horizontal_execution_merging, temporaries
from gt4py.eve import codegen


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_object import StencilObject


@backend_base.register
class DebugBackend(backend_base.BaseBackend, backend_base.CLIBackendMixin):
    """Debug backend using plain python loops."""

    name = "debug"
    options: ClassVar[dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": oir_pipeline.OirPipeline},
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
    storage_info = storage.layout.NaiveCPULayout
    languages: ClassVar[dict[str, Any]] = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = numpy_backend.ModuleGenerator

    def generate_computation(self) -> dict[str, Union[str, dict]]:
        computation_name = (
            f"{self.builder.caching.module_prefix}"
            + f"computation{self.builder.caching.module_postfix}.py"
        )

        oir = gtir_to_oir.GTIRToOIR().visit(self.builder.gtir)
        oir = horizontal_execution_merging.HorizontalExecutionMerging().visit(oir)
        oir = temporaries.LocalTemporariesToScalars().visit(oir)
        source_code = debug_codegen.DebugCodeGen().visit(oir)

        if self.builder.options.format_source:
            source_code = codegen.format_source("python", source_code)

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
