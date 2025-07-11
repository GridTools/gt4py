# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from gt4py.cartesian.backend.module_generator import BaseModuleGenerator
from gt4py.cartesian.stencil_builder import StencilBuilder


class PythonModuleGenerator(BaseModuleGenerator):
    """Module Generator for use with backends that generate python code."""

    def __init__(self, builder: StencilBuilder) -> None:
        super().__init__(builder)

    def generate_imports(self) -> str:
        caching = self.builder.caching
        comp_pkg = f"{caching.module_prefix}computation{caching.module_postfix}"

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


def recursive_write(root_path: pathlib.Path, tree: dict[str, str | dict]):
    root_path.mkdir(parents=True, exist_ok=True)

    for key, value in tree.items():
        if isinstance(value, dict):
            return recursive_write(root_path / key, value)

        src_path = root_path / key
        src_path.write_text(value)
