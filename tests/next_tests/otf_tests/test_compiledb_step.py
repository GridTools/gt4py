# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py.next.otf.compilation import build_data, cache, importer
from gt4py.next.otf.compilation.build_systems import compiledb


def test_default_compiledb_factory(compilable_source_example, clean_example_session_cache):
    otf_builder = compiledb.CompiledbFactory()(
        compilable_source_example, cache_strategy=cache.Strategy.SESSION
    )

    # make sure the example project has not been written yet
    assert not build_data.contains_data(otf_builder.root_path)

    otf_builder.build()
    data = build_data.read_data(otf_builder.root_path)

    assert data.status == build_data.BuildStatus.COMPILED
    assert pathlib.Path(otf_builder.root_path / data.module).exists()
    assert hasattr(
        importer.import_from_path(otf_builder.root_path / data.module), data.entry_point_name
    )
