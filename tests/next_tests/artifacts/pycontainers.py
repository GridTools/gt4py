# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import NamedTuple, Final, Protocol, TypeVar


import dataclasses

from gt4py import next as gtx
from gt4py.eve.extended_typing import NestedTuple, Self
from gt4py.next import common, Dimension, Field, float32, float64, Dims, containers
from gt4py.next.type_system import type_specifications as ts


TDim = Dimension("TDim")  # Meaningless dimension just for tests


class SingleElementNamedTupleContainer(NamedTuple):
    x: Field[Dims[TDim], float32]


@dataclasses.dataclass
class SingleElementDataclassContainer:
    x: Field[Dims[TDim], float32]


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
    b: ScalarsContainer
    c: tuple[tuple[NamedTupleContainer, DataclassContainer], int]


PYCONTAINERS_SAMPLES: Final[
    dict[type[containers.PyContainer], NestedTuple[common.NumericValue]]
] = {
    SingleElementNamedTupleContainer: (gtx.constructors.full({TDim: 5}, 2.0),),
    SingleElementDataclassContainer: (gtx.constructors.full({TDim: 5}, 2.0),),
    NamedTupleContainer: (
        gtx.constructors.full({TDim: 5}, 2.0),
        gtx.constructors.full({TDim: 5}, 3.0),
    ),
    DataclassContainer: (
        gtx.constructors.full({TDim: 5}, 2.0),
        gtx.constructors.full({TDim: 5}, 3.0),
    ),
    NestedDataclassContainer: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    NestedNamedTupleDataclassContainer: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    NestedDataclassNamedTupleContainer: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    NestedMixedTupleContainer: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    ScalarsContainer: (
        (1.0, 2.0),
        ((3.0, 4.0), (5.0, 6.0)),
    ),
    DeeplyNestedContainer: (
        (1.0, 2.0),
        (
            (-1.0, -2.0),
            ((-3.0, -4.0), (-5.0, -6.0)),
        ),
        (
            (
                (
                    gtx.constructors.full({TDim: 5}, 12.0),
                    gtx.constructors.full({TDim: 5}, 13.0),
                ),
                (
                    gtx.constructors.full({TDim: 5}, 22.0),
                    gtx.constructors.full({TDim: 5}, 33.0),
                ),
            ),
            3,
        ),
    ),
}


PC = TypeVar("PC", bound=containers.PyContainer)


def from_nested_tuple(container_type_hint: type[PC], data: NestedTuple) -> PC:
    """Construct a container of type `container_type_hint` from a nested tuple `data`."""

    nested_types = containers.elements_types(container_type_hint)
    keys = containers.elements_keys(container_type_hint)
    assert len(keys) == len(data), f"Expected {len(keys)} elements, got {len(data)}"

    nested_values = {
        key: from_nested_tuple(nested_type, value)
        if containers.is_container_type(nested_type := nested_types[key])
        else value
        for key, value in zip(containers.elements_keys(container_type_hint), data)
    }

    container_type = containers.container_type(container_type_hint)
    assert container_type is not None, (
        f"Type {container_type_hint} is not a supported container type"
    )

    if isinstance(keys[0], int):
        return container_type(nested_values.values())
    else:
        return container_type(**nested_values)


def to_nested_tuple(container: containers.PyContainer) -> NestedTuple:
    """Convert a container into a nested tuple."""
    return tuple(
        to_nested_tuple(value)
        if isinstance(value := getattr(container, key), containers.ANY_CONTAINER_TYPES)
        else value
        for key in containers.elements_keys(container.__class__)
    )
