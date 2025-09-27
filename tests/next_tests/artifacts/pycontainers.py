# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import NamedTuple, Final, Protocol

import dataclasses

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple
from gt4py.next import Dimension, Field, float32, float64, Dims, containers
from gt4py.next.type_system import type_specifications as ts


# Meaningless dimensions, used for tests.
TDim = Dimension("TDim")
SDim = Dimension("SDim")


class NamedTupleContainer(NamedTuple):
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]


@dataclasses.dataclass
class DataclassContainer:
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]


@dataclasses.dataclass
class NestedDataclassContainer:
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer


class NestedNamedTupleDataclassContainer(NamedTuple):
    a: DataclassContainer
    b: DataclassContainer
    c: DataclassContainer


@dataclasses.dataclass
class NestedDataclassNamedTupleContainer:
    a: NamedTupleContainer
    b: NamedTupleContainer
    c: NamedTupleContainer


@dataclasses.dataclass
class NestedMixedTupleContainer:
    a: NamedTupleContainer
    b: DataclassContainer
    c: NamedTupleContainer


@dataclasses.dataclass
class ScalarsContainer:
    a: tuple[float32, float32]
    b: tuple[tuple[float32, float32], tuple[float64, float64]]


@dataclasses.dataclass
class DeeplyNestedContainer:
    a: tuple[float32, float32]
    b: DataclassContainer
    c: ScalarsContainer
    d: tuple[tuple[NamedTupleContainer, DataclassContainer], int]


CONTAINERS_AND_VALUES: Final[
    list[tuple[NestedTuple[containers.common.NumericValue], containers.PyContainer]]
] = [
    (
        (
            x := gtx.constructors.full({TDim: 5}, 2.0),
            y := gtx.constructors.full({TDim: 5}, 3.0),
        ),
        NamedTupleContainer(x, y),
    ),
    (
        (
            x := gtx.constructors.full({TDim: 5}, 2.0),
            y := gtx.constructors.full({TDim: 5}, 3.0),
        ),
        DataclassContainer(x, y),
    ),
    (
        (
            (
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            (
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            (
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        NestedDataclassContainer(
            DataclassContainer(a_x, a_y),
            DataclassContainer(b_x, b_y),
            DataclassContainer(c_x, c_y),
        ),
    ),
    (
        (
            (
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            (
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            (
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        NestedNamedTupleDataclassContainer(
            DataclassContainer(a_x, a_y),
            DataclassContainer(b_x, b_y),
            DataclassContainer(c_x, c_y),
        ),
    ),
    (
        (
            (
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            (
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            (
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        NestedDataclassNamedTupleContainer(
            NamedTupleContainer(a_x, a_y),
            NamedTupleContainer(b_x, b_y),
            NamedTupleContainer(c_x, c_y),
        ),
    ),
    (
        (
            (
                a_x := gtx.constructors.full({TDim: 5}, 2.0),
                a_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            (
                b_x := gtx.constructors.full({TDim: 5}, 4.0),
                b_y := gtx.constructors.full({TDim: 5}, 5.0),
            ),
            (
                c_x := gtx.constructors.full({TDim: 5}, 6.0),
                c_y := gtx.constructors.full({TDim: 5}, 7.0),
            ),
        ),
        NestedMixedTupleContainer(
            NamedTupleContainer(a_x, a_y),
            DataclassContainer(b_x, b_y),
            NamedTupleContainer(c_x, c_y),
        ),
    ),
    (
        (a := ((1.0, 2.0), b := ((3.0, 4.0), (5.0, 6.0)))),
        ScalarsContainer(a, b),
    ),
    (
        (
            a := (1.0, 2.0),
            (
                b_x := gtx.constructors.full({TDim: 5}, 2.0),
                b_y := gtx.constructors.full({TDim: 5}, 3.0),
            ),
            (c_a := ((1.0, 2.0), c_b := ((3.0, 4.0), (5.0, 6.0)))),
            (
                (
                    d_0_x := gtx.constructors.full({TDim: 5}, 2.0),
                    d_0_y := gtx.constructors.full({TDim: 5}, 3.0),
                ),
                d_1 := 3,
            ),
        ),
        DeeplyNestedContainer((NamedTupleContainer(d_0_x), DataclassContainer(d_0_y)), d_1),
    ),
]
