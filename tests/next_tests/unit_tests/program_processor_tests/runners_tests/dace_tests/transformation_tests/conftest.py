# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Optional, Sequence, Union, overload, Literal, Generator

import pytest
import dace
import copy
import numpy as np
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


@pytest.fixture()
def set_dace_settings() -> Generator[None, None, None]:
    """Sets the common DaCe settings for the tests.

    The function will modify the following settings:
    - `optimizer.match_exception` exceptions during the pattern matching stage,
        especially inside `can_be_applied()` are not ignored.
    - `compiler.allow_view_arguments` allow that NumPy views can be passed to
        `CompiledSDFG` objects as arguments.
    """
    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=False)
        dace.Config.set("compiler", "allow_view_arguments", value=True)
        yield
