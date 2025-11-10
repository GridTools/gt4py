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
from gt4py.next import common, Dimension, Field, float32, float64, Dims, named_collections
from gt4py.next.type_system import type_specifications as ts


TDim = Dimension("TDim")  # Meaningless dimension just for tests


class SingleElementNamedTupleNamedCollection(NamedTuple):
    x: Field[Dims[TDim], float32]


@dataclasses.dataclass
class SingleElementDataclassNamedCollection:
    x: Field[Dims[TDim], float32]


class NamedTupleNamedCollection(NamedTuple):
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]


@dataclasses.dataclass
class DataclassNamedCollection:
    x: Field[Dims[TDim], float32]
    y: Field[Dims[TDim], float32]


@dataclasses.dataclass
class NestedDataclassNamedCollection:
    a: DataclassNamedCollection
    b: DataclassNamedCollection
    c: DataclassNamedCollection


class NestedNamedTupleDataclassNamedCollection(NamedTuple):
    a: DataclassNamedCollection
    b: DataclassNamedCollection
    c: DataclassNamedCollection


@dataclasses.dataclass
class NestedDataclassNamedTupleNamedCollection:
    a: NamedTupleNamedCollection
    b: NamedTupleNamedCollection
    c: NamedTupleNamedCollection


@dataclasses.dataclass
class NestedMixedTupleNamedCollection:
    a: NamedTupleNamedCollection
    b: DataclassNamedCollection
    c: NamedTupleNamedCollection


@dataclasses.dataclass
class ScalarsNamedCollection:
    a: tuple[float32, float32]
    b: tuple[tuple[float32, float32], tuple[float64, float64]]


@dataclasses.dataclass
class DeeplyNestedNamedCollection:
    a: tuple[float32, float32]
    b: ScalarsNamedCollection
    c: tuple[tuple[NamedTupleNamedCollection, DataclassNamedCollection], int]


PYCONTAINERS_SAMPLES: Final[
    dict[type[named_collections.CustomNamedCollection], NestedTuple[common.NumericValue]]
] = {
    SingleElementNamedTupleNamedCollection: (gtx.constructors.full({TDim: 5}, 2.0),),
    SingleElementDataclassNamedCollection: (gtx.constructors.full({TDim: 5}, 2.0),),
    NamedTupleNamedCollection: (
        gtx.constructors.full({TDim: 5}, 2.0),
        gtx.constructors.full({TDim: 5}, 3.0),
    ),
    DataclassNamedCollection: (
        gtx.constructors.full({TDim: 5}, 2.0),
        gtx.constructors.full({TDim: 5}, 3.0),
    ),
    NestedDataclassNamedCollection: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    NestedNamedTupleDataclassNamedCollection: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    NestedDataclassNamedTupleNamedCollection: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    NestedMixedTupleNamedCollection: (
        (gtx.constructors.full({TDim: 5}, 2.0), gtx.constructors.full({TDim: 5}, 3.0)),
        (gtx.constructors.full({TDim: 5}, 4.0), gtx.constructors.full({TDim: 5}, 5.0)),
        (gtx.constructors.full({TDim: 5}, 6.0), gtx.constructors.full({TDim: 5}, 7.0)),
    ),
    ScalarsNamedCollection: (
        (1.0, 2.0),
        ((3.0, 4.0), (5.0, 6.0)),
    ),
    DeeplyNestedNamedCollection: (
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


PC = TypeVar("PC", bound=named_collections.CustomNamedCollection)


def from_nested_tuple(named_collection_type_hint: type[PC], data: NestedTuple) -> PC:
    """Construct a named collection of type `named_collection_type_hint` from a nested tuple `data`."""

    nested_types = named_collections.elements_types(named_collection_type_hint)
    keys = named_collections.elements_keys(named_collection_type_hint)
    assert len(keys) == len(data), f"Expected {len(keys)} elements, got {len(data)}"

    nested_values = {
        key: from_nested_tuple(nested_type, value)
        if named_collections.is_named_collection_type(nested_type := nested_types[key])
        else value
        for key, value in zip(named_collections.elements_keys(named_collection_type_hint), data)
    }

    named_collection_type = named_collections.named_collection_type(named_collection_type_hint)
    assert named_collection_type is not None, (
        f"Type {named_collection_type_hint} is not a supported named collection type"
    )

    if isinstance(keys[0], int):
        return named_collection_type(nested_values.values())
    else:
        return named_collection_type(**nested_values)


def to_nested_tuple(named_collection: named_collections.CustomNamedCollection) -> NestedTuple:
    """Convert a named collection into a nested tuple."""
    return tuple(
        to_nested_tuple(value)
        if isinstance(
            value := getattr(named_collection, key), named_collections.NAMED_COLLECTION_TYPES
        )
        else value
        for key in named_collections.elements_keys(named_collection.__class__)
    )
