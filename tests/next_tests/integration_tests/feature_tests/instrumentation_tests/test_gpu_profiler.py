# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gt4py.next.instrumentation import gpu_profiler

from ...multi_feature_tests.ffront_tests.test_ffront_fvm_nabla import pnabla


with gpu_profiler.profile():
    pass
