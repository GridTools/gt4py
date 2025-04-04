# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re

import gt4py


def test_version():
    assert isinstance(gt4py.__version__, str)
    assert len(gt4py.__version__) and all(len(p) for p in gt4py.__version__.split("."))
    assert gt4py.__about__.__version__ == gt4py.__version__


def test_version_info():
    from packaging.version import Version

    assert isinstance(gt4py.__version_info__, Version)
    assert gt4py.__version_info__.release == tuple(
        int(p) for p in re.split("[\.\+]", gt4py.__version__)[:3]
    )
    assert gt4py.__version__.startswith(gt4py.__version_info__.public)
    assert gt4py.__version__.endswith(gt4py.__version_info__.local)
    assert gt4py.__about__.__version_info__ == gt4py.__version_info__
