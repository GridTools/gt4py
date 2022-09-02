# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


import enum
import random
import string
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Set, Type, TypeVar

from eve.concepts import (
    AnySourceLocation,
    FrozenNode,
    Node,
    SourceLocation,
    SourceLocationGroup,
    SymbolName,
    VType,
)
from eve.datamodels import Coerced
from eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from eve.type_definitions import IntEnum, StrEnum
from eve.utils import CaseStyleConverter


T = TypeVar("T")
S = TypeVar("S")


# -- Node definitions --
@enum.unique
class IntKind(IntEnum):
    """Sample int Enum."""

    MINUS = -1
    ZERO = 0
    PLUS = 1


@enum.unique
class StrKind(StrEnum):
    """Sample string Enum."""

    FOO = "foo"
    BLA = "bla"
    FIZ = "fiz"
    FUZ = "fuz"


SimpleVType = VType("simple")


class EmptyNode(Node):
    pass


class LocationNode(Node):
    loc: SourceLocation


class SimpleNode(Node):
    int_value: int
    bool_value: bool
    float_value: float
    str_value: str
    bytes_value: bytes
    int_kind: IntKind
    str_kind: StrKind


class SimpleNodeWithOptionals(Node):
    int_value: int
    float_value: Optional[float]
    str_value: Optional[str] = None


class SimpleNodeWithLoc(Node):
    int_value: int
    float_value: float
    str_value: str
    loc: Optional[AnySourceLocation]


class SimpleNodeWithCollections(Node):
    int_value: int
    int_list: List[int]
    str_set: Set[str]
    str_to_int_dict: Dict[str, int]
    loc: Optional[AnySourceLocation]


class SimpleNodeWithAbstractCollections(Node):
    int_value: int
    int_sequence: Sequence[int]
    str_set: Set[str]
    str_to_int_mapping: Mapping[str, int]
    loc: Optional[AnySourceLocation] = None


class SimpleNodeWithSymbolName(Node):
    int_value: int
    name: Coerced[SymbolName]


class SimpleNodeWithDefaultSymbolName(Node):
    int_value: int
    name: SymbolName = SymbolName("symbol_name")


class CompoundNode(Node):
    int_value: int
    location: LocationNode
    simple: SimpleNode
    simple_loc: SimpleNodeWithLoc
    simple_opt: SimpleNodeWithOptionals
    other_simple_opt: Optional[SimpleNodeWithOptionals]


class CompoundNodeWithSymbols(Node):
    int_value: int
    location: LocationNode
    simple: SimpleNode
    simple_loc: SimpleNodeWithLoc
    simple_opt: SimpleNodeWithOptionals
    other_simple_opt: Optional[SimpleNodeWithOptionals]
    node_with_name: SimpleNodeWithSymbolName


class NodeWithSymbolTable(Node, SymbolTableTrait):
    node_with_name: SimpleNodeWithSymbolName
    list_with_name: List[SimpleNodeWithSymbolName]
    node_with_default_name: SimpleNodeWithDefaultSymbolName
    compound_with_name: CompoundNodeWithSymbols


class NodeWithValidatedSymbolTable(NodeWithSymbolTable, ValidatedSymbolTableTrait):
    node_with_name: SimpleNodeWithSymbolName
    list_with_name: List[SimpleNodeWithSymbolName]
    node_with_default_name: SimpleNodeWithDefaultSymbolName
    compound_with_name: CompoundNodeWithSymbols


class FrozenSimpleNode(FrozenNode):
    int_value: int
    bool_value: bool
    float_value: float
    str_value: str
    bytes_value: bytes
    int_kind: IntKind
    str_kind: StrKind


# -- General maker functions --
_MIN_INT = -9999
_MAX_INT = 9999
_MIN_FLOAT = -999.0
_MAX_FLOAT = 999.09
_SEQUENCE_LEN = 6


def make_random_bool_value() -> bool:
    return random.choice([True, False])


def make_bool_value(*, fixed: bool = False) -> bool:
    return True if fixed else make_random_bool_value()


def make_random_int_value() -> int:
    return random.randint(_MIN_INT, _MAX_INT)


def make_int_value(*, fixed: bool = False) -> int:
    return 1 if fixed else make_random_int_value()


def make_random_neg_int_value() -> int:
    return random.randint(_MIN_INT, 1)


def make_neg_int_value(*, fixed: bool = False) -> int:
    return -2 if fixed else make_random_neg_int_value()


def make_random_pos_int_value() -> int:
    return random.randint(1, _MAX_INT)


def make_pos_int_value(*, fixed: bool = False) -> int:
    return 2 if fixed else make_random_pos_int_value()


def make_random_float_value() -> float:
    return _MIN_FLOAT + random.random() * (_MAX_FLOAT - _MIN_FLOAT)


def make_float_value(*, fixed: bool = False) -> float:
    return 1.1 if fixed else make_random_float_value()


def make_random_str_value(length: Optional[int] = None) -> str:
    length = length or _SEQUENCE_LEN
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def make_str_value(length: int = _SEQUENCE_LEN, *, fixed: bool = False) -> str:
    return string.ascii_letters[:length] if fixed else make_random_str_value(length)


def make_random_member_value(values: Sequence[T]) -> T:
    return random.choice(values)


def make_member_value(values: Sequence[T], *, fixed: bool = False) -> T:
    return values[0] if fixed else make_random_member_value(values)


def make_collection_value(
    item_type: Type[T],
    collection_type: Type[Collection[T]] = list,
    length: Optional[int] = None,
    *,
    fixed: bool = False,
) -> Collection[T]:
    length = length or _SEQUENCE_LEN

    maker_attr_name = f"make_{item_type.__name__}"
    try:
        maker = globals()[maker_attr_name]
    except Exception:

        def maker():
            return item_type()

    return collection_type([maker() for _ in range(length)])  # type: ignore


def make_mapping_value(
    key_type: Type[S],
    value_type: Type[T],
    mapping_type: Type[Mapping[S, T]] = dict,
    length: Optional[int] = None,
    *,
    fixed: bool = False,
) -> Mapping[S, T]:
    length = length or _SEQUENCE_LEN

    key_maker_attr_name = f"make_{key_type.__name__}"
    try:
        key_maker = globals()[key_maker_attr_name]
    except Exception:

        def key_maker():
            return key_type()

    value_maker_attr_name = f"make_{value_type.__name__}"
    try:
        value_maker = globals()[value_maker_attr_name]
    except Exception:

        def value_maker():
            return value_type()

    return mapping_type({key_maker(): value_maker() for _ in range(length)})  # type: ignore


def make_multinode_collection_value(
    node_class: Type[Node],
    collection_type: Type[Collection[T]] = list,
    length: Optional[int] = None,
    *,
    fixed: bool = False,
) -> Collection[T]:
    length = length or _SEQUENCE_LEN

    item_maker = globals()[
        f"make_{CaseStyleConverter.convert(node_class.__name__, 'pascal', 'snake')}"
    ]
    items = [item_maker(fixed=fixed) for _ in range(length)]

    return collection_type(items)  # type: ignore


# -- Node maker functions --
def make_source_location(*, fixed: bool = False) -> SourceLocation:
    line = make_pos_int_value(fixed=fixed)
    column = make_pos_int_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    source = f"file_{str_value}.py"

    return SourceLocation(line=line, column=column, source=source)


def make_source_location_group(*, fixed: bool = False) -> SourceLocationGroup:
    loc1 = make_source_location(fixed=fixed)
    loc2 = make_source_location(fixed=fixed)

    return SourceLocationGroup(loc1, loc2, context=make_str_value(fixed=fixed))


def make_empty_node(*, fixed: bool = False) -> LocationNode:
    return EmptyNode()


def make_location_node(*, fixed: bool = False) -> LocationNode:
    return LocationNode(loc=make_source_location(fixed=fixed))


def make_simple_node(*, fixed: bool = False) -> SimpleNode:
    int_value = make_int_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    bytes_value = make_str_value(fixed=fixed).encode()
    int_kind = IntKind.PLUS if fixed else make_member_value([*IntKind], fixed=fixed)
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind], fixed=fixed)

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_simple_node_with_optionals(*, fixed: bool = False) -> SimpleNodeWithOptionals:
    int_value = make_int_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)

    return SimpleNodeWithOptionals(int_value=int_value, float_value=float_value)


def make_simple_node_with_loc(*, fixed: bool = False) -> SimpleNodeWithLoc:
    int_value = make_int_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    loc = make_source_location(fixed=fixed)

    return SimpleNodeWithLoc(
        int_value=int_value, float_value=float_value, str_value=str_value, loc=loc
    )


def make_simple_node_with_collections(*, fixed: bool = False) -> SimpleNodeWithCollections:
    int_value = make_int_value(fixed=fixed)
    int_list = make_collection_value(int, length=3)
    str_set = make_collection_value(str, set, length=3)
    str_to_int_dict = make_mapping_value(key_type=str, value_type=int, length=3)
    loc = make_source_location(fixed=fixed)

    return SimpleNodeWithCollections(
        int_value=int_value,
        int_list=int_list,
        str_set=str_set,
        str_to_int_dict=str_to_int_dict,
        loc=loc,
    )


def make_simple_node_with_abstract_collections(
    *,
    fixed: bool = False,
) -> SimpleNodeWithAbstractCollections:
    int_value = make_int_value(fixed=fixed)
    int_sequence = make_collection_value(int, collection_type=tuple, length=3)
    str_set = make_collection_value(str, set, length=3)
    str_to_int_mapping = make_mapping_value(key_type=str, value_type=int, length=3)

    return SimpleNodeWithAbstractCollections(
        int_value=int_value,
        int_sequence=int_sequence,
        str_set=str_set,
        str_to_int_mapping=str_to_int_mapping,
    )


def make_simple_node_with_symbol_name(
    *,
    fixed: bool = False,
) -> SimpleNodeWithSymbolName:
    int_value = make_int_value(fixed=fixed)
    name = make_str_value(fixed=fixed)

    return SimpleNodeWithSymbolName(int_value=int_value, name=name)


def make_simple_node_with_default_symbol_name(
    *,
    fixed: bool = False,
) -> SimpleNodeWithDefaultSymbolName:
    int_value = make_int_value(fixed=fixed)

    return SimpleNodeWithDefaultSymbolName(int_value=int_value)


def make_compound_node(*, fixed: bool = False) -> CompoundNode:
    return CompoundNode(
        int_value=make_int_value(fixed=fixed),
        location=make_location_node(),
        simple=make_simple_node(),
        simple_loc=make_simple_node_with_loc(),
        simple_opt=make_simple_node_with_optionals(),
        other_simple_opt=None,
    )


def make_compound_node_with_symbols(*, fixed: bool = False) -> CompoundNodeWithSymbols:
    return CompoundNodeWithSymbols(
        int_value=make_int_value(fixed=fixed),
        location=make_location_node(),
        simple=make_simple_node(),
        simple_loc=make_simple_node_with_loc(),
        simple_opt=make_simple_node_with_optionals(),
        other_simple_opt=None,
        node_with_name=make_simple_node_with_symbol_name(fixed=fixed),
    )


def make_node_with_symbol_table(*, fixed: bool = False) -> NodeWithSymbolTable:
    return NodeWithSymbolTable(
        node_with_name=make_simple_node_with_symbol_name(fixed=fixed),
        node_with_default_name=make_simple_node_with_default_symbol_name(fixed=fixed),
        list_with_name=make_multinode_collection_value(
            SimpleNodeWithSymbolName, length=4, fixed=fixed
        ),
        compound_with_name=make_compound_node_with_symbols(fixed=fixed),
    )


def make_node_with_validate_symbol_table(*, fixed: bool = False) -> NodeWithValidatedSymbolTable:
    return NodeWithValidatedSymbolTable(
        node_with_name=make_simple_node_with_symbol_name(fixed=fixed),
        node_with_default_name=make_simple_node_with_default_symbol_name(fixed=fixed),
        list_with_name=make_multinode_collection_value(
            SimpleNodeWithSymbolName, length=4, fixed=fixed
        ),
        compound_with_name=make_compound_node_with_symbols(fixed=fixed),
    )


def make_frozen_simple_node(*, fixed: bool = False) -> FrozenSimpleNode:
    int_value = make_int_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    bytes_value = make_str_value(fixed=fixed).encode()
    int_kind = IntKind.PLUS if fixed else make_member_value([*IntKind], fixed=fixed)
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind], fixed=fixed)

    return FrozenSimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


# -- Makers of invalid nodes --
def make_invalid_location_node(*, fixed: bool = False) -> LocationNode:
    return LocationNode(loc=SourceLocation(line=0, column=-1, source="<str>"))


def make_invalid_at_int_simple_node(*, fixed: bool = False) -> SimpleNode:
    int_value = make_float_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    bytes_value = make_str_value(fixed=fixed).encode()
    str_value = make_str_value(fixed=fixed)
    int_kind = IntKind.PLUS if fixed else make_member_value([*IntKind], fixed=fixed)
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind], fixed=fixed)

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_float_simple_node(*, fixed: bool = False) -> SimpleNode:
    int_value = make_int_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_int_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    bytes_value = make_str_value(fixed=fixed).encode()
    int_kind = IntKind.PLUS if fixed else make_member_value([*IntKind], fixed=fixed)
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind], fixed=fixed)

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_str_simple_node(*, fixed: bool = False) -> SimpleNode:
    int_value = make_int_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    str_value = make_float_value(fixed=fixed)
    bytes_value = make_str_value(fixed=fixed).encode()
    int_kind = IntKind.PLUS if fixed else make_member_value([*IntKind])
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_bytes_simple_node(*, fixed: bool = False) -> SimpleNode:
    int_value = make_int_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    bytes_value = [1, "2", (3, 4)]
    int_kind = IntKind.PLUS if fixed else make_member_value([*IntKind])
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )


def make_invalid_at_enum_simple_node(*, fixed: bool = False) -> SimpleNode:
    int_value = make_int_value(fixed=fixed)
    bool_value = make_bool_value(fixed=fixed)
    float_value = make_float_value(fixed=fixed)
    str_value = make_str_value(fixed=fixed)
    bytes_value = make_str_value(fixed=fixed).encode()
    int_kind = "asdf"
    str_kind = StrKind.BLA if fixed else make_member_value([*StrKind])

    return SimpleNode(
        int_value=int_value,
        bool_value=bool_value,
        float_value=float_value,
        str_value=str_value,
        bytes_value=bytes_value,
        int_kind=int_kind,
        str_kind=str_kind,
    )
