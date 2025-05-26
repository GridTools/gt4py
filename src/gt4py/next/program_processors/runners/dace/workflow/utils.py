# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace

from gt4py.next import config


def set_dace_cache_config() -> None:
    """Set configuration for dace cache, shared among multiple workflow stages."""
    dace.config.Config.set("cache", value="hash")  # use the SDFG hash as cache key
    dace.config.Config.set("default_build_folder", value=str(config.BUILD_CACHE_DIR / "dace_cache"))
