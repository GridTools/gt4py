# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


from gt4py import next as gtx
from gt4py.next import fbuiltins


def test_missing_exports():
    missing = set(fbuiltins.__all__) - set(gtx.__all__)
    assert not missing, f"Missing reexports of gt4py.next.fbuiltins in gt4py.next: {missing}"
