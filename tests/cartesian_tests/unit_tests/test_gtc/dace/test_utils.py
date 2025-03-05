# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from typing import Optional

from gt4py.cartesian.gtc.common import ArithmeticOperator, DataType, CartesianOffset
from gt4py.cartesian.gtc.dace import daceir as dcir
from gt4py.cartesian.gtc.dace.constants import TASKLET_PREFIX_IN, TASKLET_PREFIX_OUT
from gt4py.cartesian.gtc.dace.utils import get_tasklet_symbol


@pytest.mark.parametrize(
    "name, is_target,offset,expected",
    [
        ("A", False, None, f"{TASKLET_PREFIX_IN}A"),
        ("A", True, None, f"{TASKLET_PREFIX_OUT}A"),
        ("A", True, CartesianOffset(i=0, j=0, k=-1), f"{TASKLET_PREFIX_OUT}Akm1"),
        ("A", False, CartesianOffset(i=1, j=-2, k=3), f"{TASKLET_PREFIX_IN}Aip1_jm2_kp3"),
        (
            "A",
            True,
            dcir.VariableKOffset(k=dcir.Literal(value="3", dtype=DataType.INT32)),
            f"{TASKLET_PREFIX_OUT}A",
        ),
    ],
)
def test_get_tasklet_symbol(
    name: str,
    is_target: bool,
    offset: Optional[CartesianOffset | dcir.VariableKOffset],
    expected: str,
) -> None:
    assert get_tasklet_symbol(name, is_target=is_target, offset=offset) == expected
