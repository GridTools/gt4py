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

    # Without keeping a reference, the module is loaded but not retained.
    n_before = len(importer._loaded_modules)
    module = importer.import_from_path(file, keep_reference=False)
    assert hasattr(module, "function")
    assert len(importer._loaded_modules) == n_before

    # Keeping a reference retains the loaded module for the process lifetime.
    kept = importer.import_from_path(file, keep_reference=True)
    assert hasattr(kept, "function")
    assert kept in importer._loaded_modules
    assert len(importer._loaded_modules) == n_before + 1
