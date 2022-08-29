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
import pytest

from functional.fencil_processors.source_modules import source_modules


def test_basic_settings_with_cpp_rejected():
    with pytest.raises(TypeError, match="Wrong language settings type"):
        source_modules.SourceModule(
            entry_point=source_modules.Function(name="basic_settings_with_cpp", parameters=[]),
            source_code="",
            library_deps=(),
            language=source_modules.Cpp,
            language_settings=source_modules.LanguageSettings(
                formatter_key="cpp", formatter_style="llvm", file_extension="cpp"
            ),
        )


def test_header_files_settings_with_cpp_accepted():
    source_modules.SourceModule(
        entry_point=source_modules.Function(name="basic_settings_with_cpp", parameters=[]),
        source_code="",
        library_deps=(),
        language=source_modules.Cpp,
        language_settings=source_modules.LanguageWithHeaderFilesSettings(
            formatter_key="cpp",
            formatter_style="llvm",
            file_extension="cpp",
            header_extension="hpp",
        ),
    )
