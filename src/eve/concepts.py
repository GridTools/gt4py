# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""Definitions of basic Eve concepts."""


from __future__ import annotations

import functools

import pydantic
import pydantic.generics

from . import iterators, utils
from .type_definitions import NOTHING, IntEnum, Str, StrEnum
from .typingx import (
    Any,
    AnyNoArgCallable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)


# -- Fields --
class ImplFieldMetadataDict(TypedDict, total=False):
    info: pydantic.fields.FieldInfo


NodeImplFieldMetadataDict = Dict[str, ImplFieldMetadataDict]


class FieldKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class FieldConstraintsDict(TypedDict, total=False):
    vtype: Union[VType, Tuple[VType, ...]]


class FieldMetadataDict(TypedDict, total=False):
    constraints: FieldConstraintsDict
    kind: FieldKind
    definition: pydantic.fields.ModelField


NodeChildrenMetadataDict = Dict[str, FieldMetadataDict]


_EVE_METADATA_KEY = "_EVE_META_"


def field(
    default: Any = NOTHING,
    *,
    default_factory: Optional[AnyNoArgCallable] = None,
    kind: Optional[FieldKind] = None,
    constraints: Optional[FieldConstraintsDict] = None,
    schema_config: Dict[str, Any] = None,
) -> pydantic.fields.FieldInfo:
    metadata = {}
    for key in ["kind", "constraints"]:
        value = locals()[key]
        if value:
            metadata[key] = value
    kwargs = schema_config or {}
    kwargs[_EVE_METADATA_KEY] = metadata

    if default is NOTHING:
        field_info = pydantic.Field(default_factory=default_factory, **kwargs)
    else:
        field_info = pydantic.Field(default, default_factory=default_factory, **kwargs)
    assert isinstance(field_info, pydantic.fields.FieldInfo)

    return field_info


in_field = functools.partial(field, kind=FieldKind.INPUT)
out_field = functools.partial(field, kind=FieldKind.OUTPUT)


# -- Models --
class Model(pydantic.BaseModel):
    class Config:
        extra = "forbid"


class FrozenModel(pydantic.BaseModel):
    class Config:
        allow_mutation = False


# -- Nodes --
_EVE_NODE_INTERNAL_SUFFIX = "__"
_EVE_NODE_IMPL_SUFFIX = "_"

AnyNode = TypeVar("AnyNode", bound="BaseNode")
ValueNode = Union[bool, bytes, int, float, str, IntEnum, StrEnum]
LeafNode = Union[AnyNode, ValueNode]
CollectionNode = Union[List[LeafNode], Dict[Any, LeafNode], Set[LeafNode]]
TreeNode = Union[AnyNode, CollectionNode]


class NodeMetaclass(pydantic.main.ModelMetaclass):
    """Custom metaclass for Node classes.

    Customize the creation of Node classes adding Eve specific attributes.

    """

    @no_type_check
    def __new__(mcls, name, bases, namespace, **kwargs):
        # Optional preprocessing of class namespace before creation:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Postprocess created class:
        # Add metadata class members
        impl_fields_metadata = {}
        children_metadata = {}
        for name, model_field in cls.__fields__.items():
            if name.endswith(_EVE_NODE_IMPL_SUFFIX):
                impl_fields_metadata[name] = {"definition": model_field}
            elif not name.endswith(_EVE_NODE_INTERNAL_SUFFIX):
                children_metadata[name] = {
                    "definition": model_field,
                    **model_field.field_info.extra.get(_EVE_METADATA_KEY, {}),
                }

        cls.__node_impl_fields__ = impl_fields_metadata
        cls.__node_children__ = children_metadata

        return cls


class BaseNode(pydantic.BaseModel, metaclass=NodeMetaclass):
    """Base class representing an IR node.

    It is currently implemented as a pydantic Model with some extra features.

    Field values should be either:

        * builtin types: `bool`, `bytes`, `int`, `float`, `str`
        * enum.Enum types
        * other :class:`Node` subclasses
        * other :class:`pydantic.BaseModel` subclasses
        * supported collections (:class:`List`, :class:`Dict`, :class:`Set`)
            of any of the previous items

    Field naming scheme:

        * Field names starting with "_" are ignored by pydantic and Eve. They
            will not be considered as `fields` and thus none of the pydantic
            features will work (type coercion, validators, etc.).
        * Field names ending with "__" are reserved for internal Eve use and
            should NOT be defined by regular users. All pydantic features will
            work on these fields anyway but they will be invisible for Eve users.
        * Field names ending with "_" are considered implementation fields
            not children nodes. They are intended to be defined by users when needed,
            typically to cache derived, non-essential information on the node.

    """

    __node_impl_fields__: ClassVar[NodeImplFieldMetadataDict]
    __node_children__: ClassVar[NodeChildrenMetadataDict]

    # Node fields
    #: Unique node-id (implementation field)
    id_: Optional[Str] = None

    @pydantic.validator("id_", pre=True, always=True)
    def _id_validator(cls: Type[AnyNode], v: Optional[str]) -> str:  # type: ignore  # validators are classmethods
        if v is None:
            v = utils.UIDGenerator.sequential_id(prefix=cls.__qualname__)
        if not isinstance(v, str):
            raise TypeError(f"id_ is not an 'str' instance ({type(v)})")
        return v

    def iter_impl_fields(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if name.endswith(_EVE_NODE_IMPL_SUFFIX) and not name.endswith(
                _EVE_NODE_INTERNAL_SUFFIX
            ):
                yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name, _ in self.__fields__.items():
            if not (
                name.endswith(_EVE_NODE_IMPL_SUFFIX) or name.endswith(_EVE_NODE_INTERNAL_SUFFIX)
            ):
                yield name, getattr(self, name)

    def iter_children_values(self) -> Generator[Any, None, None]:
        for _, node in self.iter_children():
            yield node

    def iter_tree_pre(self) -> utils.XIterator:
        return iterators.iter_tree_pre(self)

    def iter_tree_post(self) -> utils.XIterator:
        return iterators.iter_tree_post(self)

    def iter_tree_levels(self) -> utils.XIterator:
        return iterators.iter_tree_levels(self)

    iter_tree = iter_tree_pre

    class Config(Model.Config):
        pass


class GenericNode(BaseNode, pydantic.generics.GenericModel):
    pass


class Node(BaseNode):
    """Default public name for a base node class."""

    pass


class FrozenNode(Node):
    """Default public name for an inmutable base node class."""

    class Config(FrozenModel.Config):
        pass


# -- Misc --
class VType(FrozenModel):

    # VType fields
    #: Unique name
    name: Str

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
