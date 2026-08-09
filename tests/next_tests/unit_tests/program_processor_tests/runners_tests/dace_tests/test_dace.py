# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test specific features of DaCe backends."""

import pytest


dace = pytest.importorskip("dace")


# The DaCe nanobind interface replaced the `construct_arguments()` / `fast_call()`
# pair with a single `user_bind_call()`, which takes the arguments produced by the
# binding function generated for the program. There is no argument vector kept
# between calls anymore, so there is nothing left to reuse and nothing to test
# here; the binding function itself is covered by `test_dace_bindings.py`.
@pytest.mark.skip(reason="`fast_call` is no longer supported.")
def test_dace_fastcall():
    """Test reuse of SDFG arguments between program calls by means of SDFG fastcall API."""


@pytest.mark.skip(reason="`fast_call` is no longer supported.")
def test_dace_fastcall_with_connectivity():
    """Test reuse of SDFG arguments between program calls by means of SDFG fastcall API."""
