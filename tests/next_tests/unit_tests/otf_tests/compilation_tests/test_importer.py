# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import textwrap

import pytest

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

    imported_modules = set(sys.modules.keys())
    module = importer.import_from_path(file, add_to_sys_modules=False)
    assert hasattr(module, "function")
    assert "module" not in sys.modules
    assert set(sys.modules.keys()) == imported_modules

    module = importer.import_from_path(
        file, add_to_sys_modules=True, sys_modules_prefix="_temp_test_prefix_"
    )
    assert hasattr(module, "function")
    assert len(sys.modules.keys()) == len(imported_modules) + 1
    assert "_temp_test_prefix_.module" in sys.modules

    with pytest.raises(
        ValueError, match="Cannot use 'sys_modules_prefix' if 'add_to_sys_modules' is False"
    ):
        importer.import_from_path(
            file, add_to_sys_modules=False, sys_modules_prefix="_temp_test_prefix_"
        )
