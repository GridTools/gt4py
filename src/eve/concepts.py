# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

"""Definitions of basic Eve."""


from __future__ import annotations

import collections.abc
import functools

import pydantic
import pydantic.generics

from . import utils
from .type_definitions import NOTHING, Enum, IntEnum, Str, StrEnum
from .typingx import (
    Any,
    AnyNoArgCallable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    no_type_check,
)


try:
    # For perfomance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz  # noqa: F401  # imported but unused


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
            if not name.endswith(_EVE_NODE_INTERNAL_SUFFIX):
                if name.endswith(_EVE_NODE_IMPL_SUFFIX):
                    impl_fields_metadata[name] = {"definition": model_field}
                else:
                    children_metadata[name] = {
                        "definition": model_field,
                        **model_field.field_info.extra.get(_EVE_METADATA_KEY, {}),
                    }

        cls.__node_impl_fields__ = impl_fields_metadata
        cls.__node_children__ = children_metadata

        return cls


KeyValue = Tuple[Union[int, str], TreeNode]
TreeIterationItem = Union[TreeNode, Tuple[KeyValue, TreeNode]]


def generic_iter_children(
    node: TreeNode, *, with_keys: bool = False
) -> Iterable[Union[TreeNode, Tuple[KeyValue, TreeNode]]]:
    """Create an iterator to traverse values as Eve tree nodes.

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    if isinstance(node, BaseNode):
        return node.iter_children() if with_keys else node.iter_children_values()
    elif isinstance(node, (list, tuple)) or (
        isinstance(node, collections.abc.Sequence) and not isinstance(node, (str, bytes))
    ):
        return enumerate(node) if with_keys else iter(node)
    elif isinstance(node, (set, collections.abc.Set)):
        return zip(node, node) if with_keys else iter(node)  # type: ignore  # problems with iter(Set)
    elif isinstance(node, (dict, collections.abc.Mapping)):
        return node.items() if with_keys else node.values()

    return iter(())


class TraversalOrder(Enum):
    PRE_ORDER = "pre"
    POST_ORDER = "post"
    LEVELS_ORDER = "levels"


def _iter_tree_pre(
    node: TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
) -> Generator[TreeIterationItem, None, None]:
    """Create a pre-order tree traversal iterator (Depth-First Search).

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    if with_keys:
        yield __key__, node
        for key, child in generic_iter_children(node, with_keys=True):
            yield from _iter_tree_pre(child, with_keys=True, __key__=key)
    else:
        yield node
        for child in generic_iter_children(node, with_keys=False):
            yield from _iter_tree_pre(child, with_keys=False)


def _iter_tree_post(
    node: TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
) -> Generator[TreeIterationItem, None, None]:
    """Create a post-order tree traversal iterator (Depth-First Search).

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    if with_keys:
        for key, child in generic_iter_children(node, with_keys=True):
            yield from _iter_tree_post(child, with_keys=True, __key__=key)
        yield __key__, node
    else:
        for child in generic_iter_children(node, with_keys=False):
            yield from _iter_tree_post(child, with_keys=False)
        yield node


def _iter_tree_levels(
    node: TreeNode,
    *,
    with_keys: bool = False,
    __key__: Optional[Any] = None,
    __queue__: Optional[List] = None,
) -> Generator[TreeIterationItem, None, None]:
    """Create a tree traversal iterator by levels (Breadth-First Search).

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    __queue__ = __queue__ or []
    if with_keys:
        yield __key__, node
        __queue__.extend(generic_iter_children(node, with_keys=True))
        if __queue__:
            key, child = __queue__.pop(0)
            yield from _iter_tree_levels(child, with_keys=True, __key__=key, __queue__=__queue__)
    else:
        yield node
        __queue__.extend(generic_iter_children(node, with_keys=False))
        if __queue__:
            child = __queue__.pop(0)
            yield from _iter_tree_levels(child, with_keys=False, __queue__=__queue__)


iter_tree_pre = utils.as_xiter(_iter_tree_pre)
iter_tree_post = utils.as_xiter(_iter_tree_post)
iter_tree_levels = utils.as_xiter(_iter_tree_levels)


def iter_tree(
    node: TreeNode,
    traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER,
    *,
    with_keys: bool = False,
) -> utils.XIterable[TreeIterationItem]:
    """Create a tree traversal iterator.

    Args:
        traversal_order: Tree nodes traversal order.

        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    if traversal_order is traversal_order.PRE_ORDER:
        return iter_tree_pre(node=node, with_keys=with_keys)
    elif traversal_order is traversal_order.POST_ORDER:
        return iter_tree_post(node=node, with_keys=with_keys)
    elif traversal_order is traversal_order.LEVELS_ORDER:
        return iter_tree_levels(node=node, with_keys=with_keys)
    else:
        raise ValueError(f"Invalid '{traversal_order}' traversal order.")


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

    def iter_impl_fields(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__node_impl_fields__.keys():
            yield name, getattr(self, name)

    def iter_children(self) -> Generator[Tuple[str, Any], None, None]:
        for name in self.__node_children__.keys():
            yield name, getattr(self, name)

    def iter_children_values(self) -> Generator[Any, None, None]:
        for name in self.__node_children__.keys():
            yield getattr(self, name)

    def iter_tree_pre(self) -> utils.XIterable:
        return iter_tree_pre(self)

    def iter_tree_post(self) -> utils.XIterable:
        return iter_tree_post(self)

    def iter_tree_levels(self) -> utils.XIterable:
        return iter_tree_levels(self)

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
