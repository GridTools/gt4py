# GT4Py Project - GridTools Framework
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
import dataclasses
import pathlib
from typing import Callable, Generic, TypeVar

from functional.otf import step_types
from functional.program_processors.builders import build_data, cache, importer
from functional.program_processors.source_modules import source_modules


SL = TypeVar("SL")
ST = TypeVar("ST")


def is_compiled(data: build_data.OTFBuildData) -> bool:
    return data.status >= build_data.OTFBuildStatus.COMPILED


def module_exists(data: build_data.OTFBuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


@dataclasses.dataclass(frozen=True)
class OnTheFlyCompiler(Generic[SL, ST]):
    cache_strategy: cache.Strategy
    builder_factory: step_types.BuildSystemProjectGenerator[SL, ST, source_modules.Python]
    force_recompile: bool = False

    def __call__(
        self, inp: source_modules.OTFSourceModule[SL, ST, source_modules.Python]
    ) -> Callable:
        src_dir = cache.get_cache_folder(inp, self.cache_strategy)

        data = build_data.read_data(src_dir)

        if not data or not is_compiled(data) or self.force_recompile:
            self.builder_factory(inp, self.cache_strategy).build()

        new_data = build_data.read_data(src_dir)

        if not is_compiled(new_data) or not module_exists(new_data, src_dir):
            raise CompilerError(
                "On-the-fly compilation unsuccessful for {inp.source_module.entry_point.name}!"
            )

        return getattr(
            importer.import_from_path(src_dir / new_data.module), new_data.entry_point_name
        )


class CompilerError(Exception):
    ...
