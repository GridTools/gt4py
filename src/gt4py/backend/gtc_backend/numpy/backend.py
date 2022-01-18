# -*- coding: utf-8 -*-
#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
from gtc.numpy import npir
from gtc.numpy.npir_codegen import NpirCodegen
from gtc.numpy.oir_to_npir import OirToNpir
from gtc.passes.oir_optimizations.caches import (
    FillFlushToLocalKCaches,
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_pipeline import DefaultPipeline, OirPipeline


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
    options: ClassVar[Dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
        # TODO: Implement this option in source code
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
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

        source = NpirCodegen.apply(self.npir)
        if self.builder.options.format_source:
            source = format_source("python", source)

        return {computation_name: source}

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
        base_oir = GTIRToOIR().visit(self.builder.gtir)
        oir_pipeline = self.builder.options.backend_opts.get(
            "oir_pipeline",
            DefaultPipeline(
                skip=[
                    IJCacheDetection,
                    KCacheDetection,
                    PruneKCacheFills,
                    PruneKCacheFlushes,
                    FillFlushToLocalKCaches,
                ]
            ),
        )
        oir = oir_pipeline.run(base_oir)
        return OirToNpir().visit(oir)

    @property
    def npir(self) -> npir.Computation:
        key = "gtcnumpy:npir"
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_npir()})
        return self.builder.backend_data[key]
