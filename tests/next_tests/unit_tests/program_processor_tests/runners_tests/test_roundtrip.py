# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

from gt4py.next.program_processors.runners import roundtrip


@pytest.mark.parametrize("debug_order", [(False, True), (True, False)])
def test_load_module_caches_by_debug_flag(debug_order, monkeypatch):
    monkeypatch.setattr(roundtrip, "_MODULE_CACHE", {})
    source_code = "VALUE = 42\n"

    modules = {debug: roundtrip._load_module(source_code, debug) for debug in debug_order}
    try:
        assert modules[False] is not modules[True]
        assert modules[False].VALUE == modules[True].VALUE == 42
        # Only the debug module is backed by a real '.py' file.
        assert not hasattr(modules[False], "__file__")
        assert modules[True].__file__.endswith(".py")
        assert os.path.isfile(modules[True].__file__)

        for debug in debug_order:
            assert roundtrip._load_module(source_code, debug) is modules[debug]
    finally:
        os.remove(modules[True].__file__)
