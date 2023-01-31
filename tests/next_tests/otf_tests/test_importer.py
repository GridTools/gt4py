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

import textwrap

from gt4py.next.otf.compilation import importer


def test_import_from_path(tmp_path):
    src_module = textwrap.dedent(
        """\
    def function(a, b):
        return a + b
    """
    )
    file = tmp_path / "module.py"
    file.write_text(src_module, "utf-8")
    module = importer.import_from_path(file)
    assert hasattr(module, "function")
