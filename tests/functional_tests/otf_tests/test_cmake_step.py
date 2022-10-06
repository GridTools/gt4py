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
import pathlib

from functional.otf.compile import build_data
from functional.otf.compile.build_systems import cmake
from functional.program_processors.builders import cache, importer


def test_default_builder_generator(compilable_source_example, clean_example_session_cache):
    otf_builder = cmake.make_cmake_factory()(
        source=compilable_source_example, cache_strategy=cache.Strategy.SESSION
    )
    assert not build_data.contains_data(otf_builder.root_path)

    otf_builder.build()

    data = build_data.read_data(otf_builder.root_path)

    assert data.status == build_data.OTFBuildStatus.COMPILED
    assert pathlib.Path(otf_builder.root_path / data.module).exists()
    assert hasattr(
        importer.import_from_path(otf_builder.root_path / data.module), data.entry_point_name
    )
