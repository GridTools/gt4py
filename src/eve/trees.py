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

"""Iterator utils."""


from __future__ import annotations

import abc
import collections.abc
import functools

from . import utils
from .extended_typing import Any, Iterable, List, Optional, Protocol, Tuple, Type, TypeVar, Union
from .type_definitions import Enum


try:
    # For perfomance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz  # noqa: F401  # imported but unused


TreeKey = Union[int, str]


class TreeLike(abc.ABC):
    ...


class Tree(Protocol):
    @abc.abstractmethod
    def iter_node_values(self) -> Iterable:
        ...

    @abc.abstractmethod
    def iter_node_items(self) -> Iterable[Tuple[TreeKey, Any]]:
        ...


TreeLike.register(Tree)


# -- Node iteration --
@functools.singledispatch
def iter_node_values(node: TreeLike) -> Iterable:
    """Create an iterator to traverse values as Eve tree nodes."""
    return node.iter_node_values() if hasattr(node, "iter_node_values") else iter(())


@functools.singledispatch
def iter_node_items(node: TreeLike) -> Iterable[Tuple[TreeKey, Any]]:
    """Create an iterator to traverse values as Eve tree nodes."""
    return node.iter_node_items() if hasattr(node, "iter_node_items") else iter(())


_T = TypeVar("_T")


def register_tree_type(*types: Type[_T], iter_values_fn, iter_items_fn) -> Type[_T]:
    for t in types:
        TreeLike.register(t)
        iter_node_values.register(t)(iter_values_fn)
        iter_node_items.register(t)(iter_items_fn)


register_tree_type(str, bytes, iter_values_fn=lambda _: iter(()), iter_items_fn=lambda _: iter(()))

register_tree_type(
    collections.abc.Sequence,
    collections.abc.Set,
    iter_values_fn=lambda x: iter(x),
    iter_items_fn=lambda x: enumerate(x),
)

register_tree_type(
    collections.abc.Sequence,
    collections.abc.Set,
    iter_values_fn=lambda x: iter(x),
    iter_items_fn=lambda x: enumerate(x),
)
register_tree_type(
    collections.abc.Mapping,
    iter_values_fn=lambda x: x.values(),
    iter_items_fn=lambda x: x.items(),
)


# -- Tree traversals --
class TraversalOrder(Enum):
    PRE_ORDER = "pre"
    POST_ORDER = "post"
    BFS_ORDER = "bfs"


def _pre_walk_items(
    node: TreeLike, *, __key__: Optional[TreeKey] = None
) -> Iterable[Tuple[TreeKey, Any]]:
    """Create a pre-order tree traversal iterator of (key, value) pairs."""
    yield __key__, node
    for key, child in iter_node_items(node):
        yield from _pre_walk_items(child, __key__=key)


def _pre_walk_values(node: TreeLike) -> Iterable[Tuple[Any]]:
    """Create a pre-order tree traversal iterator of values."""
    yield node
    for child in iter_node_values(node):
        yield from _pre_walk_values(child)


pre_walk_items = utils.as_xiter(_pre_walk_items)
pre_walk_values = utils.as_xiter(_pre_walk_values)


def _post_walk_items(
    node: TreeLike, *, __key__: Optional[TreeKey] = None
) -> Iterable[Tuple[TreeKey, Any]]:
    """Create a post-order tree traversal iterator of (key, value) pairs."""
    yield __key__, node
    for key, child in iter_node_items(node):
        yield from _post_walk_items(child, __key__=key)


def _post_walk_values(node: TreeLike) -> Iterable[Tuple[Any]]:
    """Create a post-order tree traversal iterator of values."""
    if (iter_node_values := getattr(node, "iter_node_values", None)) is not None:
        for child in iter_node_values():
            yield from _post_walk_values(child)
    yield node


post_walk_items = utils.as_xiter(_post_walk_items)
post_walk_values = utils.as_xiter(_post_walk_values)


def _bfs_walk_items(
    node: TreeLike, *, __key__: Optional[Any] = None, __queue__: Optional[List] = None
) -> Iterable[Tuple[TreeKey, Any]]:
    """Create a tree traversal iterator of (key, value) pairs by tree levels (Breadth-First Search)."""
    __queue__ = __queue__ or []
    yield __key__, node
    __queue__.extend(iter_node_items(node))
    if __queue__:
        key, child = __queue__.pop(0)
        yield from _bfs_walk_items(child, __key__=key, __queue__=__queue__)


def _bfs_walk_values(
    node: TreeLike, *, __queue__: Optional[List] = None
) -> Iterable[Tuple[TreeKey, Any]]:
    """Create a tree traversal iterator of values by tree levels (Breadth-First Search)."""
    __queue__ = __queue__ or []
    yield node
    if (iter_node_values := getattr(node, "iter_node_values", None)) is not None:
        __queue__.extend(iter_node_values())
    if __queue__:
        child = __queue__.pop(0)
        yield from _bfs_walk_values(child, __queue__=__queue__)


bfs_walk_items = utils.as_xiter(_bfs_walk_items)
bfs_walk_values = utils.as_xiter(_bfs_walk_values)


def walk_items(
    node: TreeLike, traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER
) -> utils.XIterable[Tuple[TreeKey, Any]]:
    """Create a tree traversal iterator of (key, value) pairs."""
    if traversal_order is traversal_order.PRE_ORDER:
        return pre_walk_items(node=node)
    elif traversal_order is traversal_order.POST_ORDER:
        return post_walk_items(node=node)
    elif traversal_order is traversal_order.BFS_ORDER:
        return bfs_walk_items(node=node)
    else:
        raise ValueError(f"Invalid '{traversal_order}' traversal order.")


def walk_values(
    node: TreeLike, traversal_order: TraversalOrder = TraversalOrder.PRE_ORDER
) -> utils.XIterable[Any]:
    """Create a tree traversal iterator of values."""
    if traversal_order is traversal_order.PRE_ORDER:
        return pre_walk_values(node=node)
    elif traversal_order is traversal_order.POST_ORDER:
        return post_walk_values(node=node)
    elif traversal_order is traversal_order.BFS_ORDER:
        return bfs_walk_values(node=node)
    else:
        raise ValueError(f"Invalid '{traversal_order}' traversal order.")
