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

import pytest

from gt4py.next.otf import languages, stages
from gt4py.next.otf.binding import interface


def test_basic_settings_with_cpp_rejected():
    with pytest.raises(TypeError, match="Wrong language settings type"):
        stages.ProgramSource(
            entry_point=interface.Function(name="basic_settings_with_cpp", parameters=[]),
            source_code="",
            library_deps=(),
            language=languages.Cpp,
            language_settings=languages.LanguageSettings(
                formatter_key="cpp", formatter_style="llvm", file_extension="cpp"
            ),
        )


def test_header_files_settings_with_cpp_accepted():
    stages.ProgramSource(
        entry_point=interface.Function(name="basic_settings_with_cpp", parameters=[]),
        source_code="",
        library_deps=(),
        language=languages.Cpp,
        language_settings=languages.LanguageWithHeaderFilesSettings(
            formatter_key="cpp",
            formatter_style="llvm",
            file_extension="cpp",
            header_extension="hpp",
        ),
    )
