# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from .fixtures.common import uids
from .fixtures.compilation import isolate_ongoing_compilations


def pytest_configure(config):
    # gt4py's own tests fail loudly on a `DimensionRef` compared with a dimension class: such a
    # comparison is always a missed conversion at the Python edge (off by default for users).
    from gt4py.next import config as gtx_config

    gtx_config.STRICT_DIMENSION_REF_COMPARISON = True
