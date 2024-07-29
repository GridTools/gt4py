# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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


@pytest.fixture(autouse=True)
def _set_dace_settings() -> Generator[None, None, None]:
    """Enables the correct settings in DaCe."""
    with dace.config.temporary_config():
        dace.Config.set("optimizer", "match_exception", value=True)
        yield
