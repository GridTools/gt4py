# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Type, Union

from gt4py import storage
from gt4py.cartesian.backend.base import BaseBackend, CLIBackendMixin, register
from gt4py.cartesian.backend.numpy_backend import ModuleGenerator
from gt4py.cartesian.gtc.debug.debug_codegen import DebugCodeGen
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.cartesian.gtc.passes.oir_pipeline import OirPipeline
from gt4py.eve.codegen import format_source


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_object import StencilObject


def recursive_write(root_path: Path, tree: dict[str, Union[str, dict]]):
    root_path.mkdir(parents=True, exist_ok=True)
    for key, value in tree.items():
        if isinstance(value, dict):
            recursive_write(root_path / key, value)
        else:
            src_path = root_path / key
            src_path.write_text(value)


@register
class DebugBackend(BaseBackend, CLIBackendMixin):
    """Debug backend using plain python loops."""

    name = "debug"
    options: ClassVar[dict[str, Any]] = {
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
        # TODO: Implement this option in source code
        "ignore_np_errstate": {"versioning": True, "type": bool},
    }
    storage_info = storage.layout.NaiveCPULayout
    languages = {"computation": "python", "bindings": ["python"]}
    MODULE_GENERATOR_CLASS = ModuleGenerator

    def generate_computation(self) -> dict[str, Union[str, dict]]:
        computation_name = (
            self.builder.caching.module_prefix
            + "computation"
            + self.builder.caching.module_postfix
            + ".py"
        )
        oir = GTIRToOIR().visit(self.builder.gtir)
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
            recursive_write(src_dir, self.generate_computation())
        return self.make_module()
