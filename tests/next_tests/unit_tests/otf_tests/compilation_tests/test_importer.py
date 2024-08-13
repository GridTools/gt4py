# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
