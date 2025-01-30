# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

from gt4py.next import config
from gt4py.next.otf.compilation import build_data, cache, importer
from gt4py.next.otf.compilation.build_systems import cmake


def test_default_cmake_factory(compilable_source_example, clean_example_session_cache):
    otf_builder = cmake.CMakeFactory()(
        source=compilable_source_example, cache_lifetime=config.BuildCacheLifetime.SESSION
    )
    assert not build_data.contains_data(otf_builder.root_path)

    otf_builder.build()

    data = build_data.read_data(otf_builder.root_path)

    assert data.status == build_data.BuildStatus.COMPILED
    assert pathlib.Path(otf_builder.root_path / data.module).exists()
    assert hasattr(
        importer.import_from_path(otf_builder.root_path / data.module), data.entry_point_name
    )
