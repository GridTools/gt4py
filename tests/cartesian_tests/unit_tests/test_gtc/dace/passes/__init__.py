# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# Skip this entire folder when we collecting tests and "dace" is not installed as a dependency.
pytest.importorskip("dace")
