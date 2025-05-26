# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import operator
from typing import Generator, Optional, Type

import boltons.typeutils

from gt4py import eve
from gt4py.cartesian import utils as gt_utils
from gt4py.cartesian.frontend.nodes import Location, Node


#: Marker value used to avoid confusion with `None`
#: (specially in contexts where `None` could be a valid value)
NOTHING = boltons.typeutils.make_sentinel(name="NOTHING", var_name="NOTHING")


def iter_attributes(node: Node):
    """
    Yield a tuple of ``(attrib_name, value)`` for each attribute in ``node.attributes``
    that is present on *node*.
    """
    for attrib_name in node.attributes:
        try:
            yield attrib_name, getattr(node, attrib_name)
        except AttributeError:
            pass


class IRNodeVisitor:
    def visit(self, node: Node, **kwargs):
        return self._visit(node, **kwargs)

    def _visit(self, node: Node, **kwargs):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: Node, **kwargs):
        items = []
        if isinstance(node, (str, bytes, bytearray)):
            pass
        elif isinstance(node, collections.abc.Mapping):
            items = node.items()
        elif isinstance(node, collections.abc.Iterable):
            items = enumerate(node)
        elif isinstance(node, Node):
            items = iter_attributes(node)
        else:
            pass

        for _, value in items:
            self._visit(value, **kwargs)


class IRNodeMapper:
    def visit(self, node: Node, **kwargs):
        return self._visit(node, **kwargs)

    def _visit(self, node: Node, **kwargs):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: Node, **kwargs):
        if isinstance(node, (str, bytes, bytearray)):
            return node
        elif isinstance(node, collections.abc.Iterable):
            if isinstance(node, collections.abc.Mapping):
                items = node.items()
            else:
                items = enumerate(node)
            setattr_op = operator.setitem
            delattr_op = operator.delitem
        elif isinstance(node, Node):
            items = iter_attributes(node)
            setattr_op = setattr
            delattr_op = delattr
        else:
            return node

        del_items = []
        for key, old_value in items:
            new_value = self._visit(old_value, **kwargs)
            if new_value == NOTHING:
                del_items.append(key)
            elif new_value != old_value:
                setattr_op(node, key, new_value)
        for key in reversed(del_items):  # reversed, so that keys remain valid in sequences
            delattr_op(node, key)

        return node


def iter_nodes_of_type(root_node: Node, node_type: Type) -> Generator[Node, None, None]:
    """Yield an iterator over the nodes of node_type inside root_node in DFS order."""

    def recurse(node: Node) -> Generator[Node, None, None]:
        for _, value in iter_attributes(node):
            if isinstance(node, collections.abc.Iterable):
                if isinstance(node, collections.abc.Mapping):
                    children = node.values()
                else:
                    children = node
            else:
                children = gt_utils.listify(value)

            for value in children:
                if isinstance(value, Node):
                    yield from recurse(value)

            if isinstance(node, node_type):
                yield node

    yield from recurse(root_node)


def location_to_source_location(loc: Optional[Location]) -> Optional[eve.SourceLocation]:
    if loc is None or loc.line <= 0 or loc.column <= 0:
        return None
    return eve.SourceLocation(line=loc.line, column=loc.column, filename=loc.scope)
