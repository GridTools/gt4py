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
from gt4py.cartesian.gtc import numpy, passes
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.cartesian.gtc.numpy import npir
from gt4py.cartesian.gtc.passes import oir_optimizations as oir_opt
from gt4py.eve import codegen
from gt4py.storage import layout


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_object import StencilObject


@backend.register
class NumpyBackend(backend.BaseBackend):
    """NumPy backend using gtc."""

    name = "numpy"
    options: ClassVar[dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": passes.OirPipeline},
        # TODO: Implement this option in source code
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
    storage_info = layout.NaiveCPULayout
    languages: ClassVar[dict] = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = py_common.PythonModuleGenerator

    def generate_computation(self) -> dict[str, str | dict]:
        ignore_np_errstate = self.builder.options.backend_opts.get("ignore_np_errstate", True)
        source = numpy.NpirCodegen.apply(self.npir, ignore_np_errstate=ignore_np_errstate)

        if self.builder.options.format_source:
            source = codegen.format_source("python", source)

        caching = self.builder.caching
        computation_name = f"{caching.module_prefix}computation{caching.module_postfix}.py"
        return {computation_name: source}

    def generate(self) -> type[StencilObject]:
        self.check_options(self.builder.options)
        src_dir = self.builder.module_path.parent

        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            py_common.recursive_write(src_dir, self.generate_computation())

        return self.make_module()

    def _make_npir(self) -> npir.Computation:
        base_oir = GTIRToOIR().visit(self.builder.gtir)
        oir_pipeline = self.builder.options.backend_opts.get(
            "oir_pipeline",
            passes.DefaultPipeline(
                skip=[
                    oir_opt.IJCacheDetection,
                    oir_opt.KCacheDetection,
                    oir_opt.PruneKCacheFills,
                    oir_opt.PruneKCacheFlushes,
                ]
            ),
        )
        oir_node = oir_pipeline.run(base_oir)
        base_npir = numpy.OirToNpir().visit(oir_node)
        return numpy.ScalarsToTemporaries().visit(base_npir)

    @property
    def npir(self) -> npir.Computation:
        key = "gtcnumpy:npir"
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_npir()})
        return self.builder.backend_data[key]
