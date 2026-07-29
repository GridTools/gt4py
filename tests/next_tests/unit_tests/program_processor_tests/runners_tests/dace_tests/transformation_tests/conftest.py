# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def common_dace_config() -> Generator[None, None, None]:
    """Sets the common DaCe settings for the tests.

    The function will modify the following settings:
    - `optimizer.match_exception` exceptions during the pattern matching stage,
        especially inside `can_be_applied()` are not ignored.
    - `compiler.allow_view_arguments` allow that NumPy views can be passed to
        `CompiledSDFG` objects as arguments.
    """
    import dace

    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=True)
        dace.Config.set("compiler", "allow_view_arguments", value=True)

        # If the cache is used, then only the name is considered. Thus in tests
        #  that compile the SDFG multiple times, for example to check if the result
        #  is the same after the transformation, will only compile the SDFG once.
        #  Thus potential errors by the transformation are hidden.
        dace.Config.set("compiler.use_cache", value=False)

        yield
