# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.otf import languages, stages
from gt4py.next.otf.binding import interface


def test_header_files_settings_with_cpp_accepted():
    stages.ProgramSource(
        entry_point=interface.Function(name="basic_settings_with_cpp", parameters=[]),
        source_code="",
        library_deps=(),
        language_settings=languages.CPPLanguageSettings(),
    )
