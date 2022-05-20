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

"""Iterator utils."""


from __future__ import annotations

import collections.abc

from . import concepts, utils
from .type_definitions import Enum
from .typingx import Any, Generator, Iterable, List, Optional, Tuple, Union


try:
    # For perfomance reasons, try to use cytoolz when possible (using cython)
    import cytoolz as toolz
except ModuleNotFoundError:
    # Fall back to pure Python toolz
    import toolz  # noqa: F401  # imported but unused


KeyValue = Tuple[Union[int, str], Any]
TreeIterationItem = Union[Any, Tuple[KeyValue, Any]]


def generic_iter_children(
    node: concepts.TreeNode, *, with_keys: bool = False
) -> Iterable[Union[Any, Tuple[KeyValue, Any]]]:
    """Create an iterator to traverse values as Eve tree nodes.

    Args:
        with_keys: Return tuples of (key, object) values where keys are
            the reference to the object node in the parent.
            Defaults to `False`.

    """
    if isinstance(node, concepts.BaseNode):
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
    node: concepts.TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
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
    node: concepts.TreeNode, *, with_keys: bool = False, __key__: Optional[Any] = None
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
    node: concepts.TreeNode,
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
    node: concepts.TreeNode,
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
