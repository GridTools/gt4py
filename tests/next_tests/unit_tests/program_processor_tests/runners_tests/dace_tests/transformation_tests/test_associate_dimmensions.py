# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import copy

from typing import Sequence, overload, Literal

dace = pytest.importorskip("dace")
from dace import subsets as dace_sbs

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)


@overload
def _perform_test(
    sbs1: str,
    sbs2: str,
    dmap: dict[int, int],
    drop1: Sequence[int],
    drop2: Sequence[int],
    allow_trivail_dimensions: bool = True,
    expect_to_fail: Literal[False] = False,
) -> None: ...


@overload
def _perform_test(
    sbs1: str,
    sbs2: str,
    dmap: None,
    drop1: None,
    drop2: None,
    expect_to_fail: Literal[True],
) -> None: ...


def _perform_test(
    sbs1: str,
    sbs2: str,
    dmap: dict[int, int],
    drop1: Sequence[int],
    drop2: Sequence[int],
    allow_trivail_dimensions: bool = True,
    expect_to_fail: bool = False,
) -> None:
    try:
        res_dmap, res_drop1, res_drop2 = gtx_transformations.utils.associate_dimmensions(
            sbs1=dace_sbs.Range.from_string(sbs1),
            sbs2=dace_sbs.Range.from_string(sbs2),
            allow_trivail_dimensions=allow_trivail_dimensions,
        )

    except ValueError:
        assert expect_to_fail, f"Expected not to fail, but failed."
        return

    assert res_dmap == dmap, f"Expected dimension mapping '{dmap}' but got '{res_dmap}'"
    assert res_drop1 == drop1, (
        f"Expected to drop dimensions '{drop1}' of first subset, but '{drop1}' were selected"
    )
    assert res_drop2 == drop2, (
        f"Expected to drop dimensions '{drop2}' of second subset, but '{drop2}' were selected"
    )


def test_associate_dim_1():
    _perform_test(
        sbs1="0:10, 1",
        sbs2="0:10",
        dmap={0: 0},
        drop1=[1],
        drop2=[],
    )


def test_associate_dim_2():
    _perform_test(
        sbs1="0:10",
        sbs2="0:10, 1",
        dmap={0: 0},
        drop1=[],
        drop2=[1],
    )


def test_associate_dim_3():
    _perform_test(
        sbs1="1, 0:10",
        sbs2="0:10",
        dmap={1: 0},
        drop1=[0],
        drop2=[],
    )


def test_associate_dim_4():
    _perform_test(
        sbs1="0:10",
        sbs2="1, 0:10",
        dmap={0: 1},
        drop1=[],
        drop2=[0],
    )


def test_associate_dim_5():
    _perform_test(
        sbs1="__i, 0:5, 1, 0:10",
        sbs2="1, 0:5, 0:10, 1, __j",
        dmap={0: 0, 1: 1, 3: 2},
        drop1=[2],
        drop2=[3, 4],
        allow_trivail_dimensions=True,
    )


def test_associate_dim_6():
    _perform_test(
        sbs1="__i, 0:5, 1, 0:10",
        sbs2="1, 0:5, 0:10, 1, __j",
        dmap={1: 1, 3: 2},
        drop1=[0, 2],
        drop2=[0, 3, 4],
        allow_trivail_dimensions=False,
    )


def test_associate_dim_7():
    _perform_test(
        sbs1="__i, 0:5, 1, 0:10, 0:2",
        sbs2="1, 0:5, 0:10, 1, __j",
        dmap=None,
        drop1=None,
        drop2=None,
        expect_to_fail=True,
    )
