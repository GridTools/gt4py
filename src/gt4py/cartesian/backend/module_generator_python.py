# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.backend.module_generator import BaseModuleGenerator
from gt4py.cartesian.stencil_builder import StencilBuilder


class PythonModuleGenerator(BaseModuleGenerator):
    """Module Generator for use with backends that generate python code."""

    def __init__(self, builder: StencilBuilder) -> None:
        super().__init__(builder)

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
