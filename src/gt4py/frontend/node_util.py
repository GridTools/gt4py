# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
import collections
import copy
import operator
from typing import Generator, Optional, Type

import gtc.utils as gtc_utils
from gt4py import utils as gt_utils
from gtc import common

from .nodes import Location, Node


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

        for key, value in items:
            self._visit(value, **kwargs)


class IRNodeInspector:
    def visit(self, node: Node):
        return self._visit((), None, node)

    def _visit(self, path: tuple, node_name: str, node):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(path, node_name, node)

    def generic_visit(self, path: tuple, node_name: str, node: Node):
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

        for key, value in items:
            self._visit((*path, node_name), key, value)


class IRNodeMapper:
    def visit(self, node: Node):
        keep_node, new_node = self._visit((), None, node)
        return new_node if keep_node else None

    def _visit(self, path: tuple, node_name: str, node: Node):
        visitor = self.generic_visit
        if isinstance(node, Node):
            for node_class in node.__class__.__mro__:
                method_name = "visit_" + node_class.__name__
                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

        return visitor(path, node_name, node)

    def generic_visit(self, path: tuple, node_name: str, node: Node):
        if isinstance(node, (str, bytes, bytearray)):
            return True, node
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
            return True, node

        del_items = []
        for key, old_value in items:
            keep_item, new_value = self._visit((*path, node_name), key, old_value)
            if not keep_item:
                del_items.append(key)
            elif new_value != old_value:
                setattr_op(node, key, new_value)
        for key in reversed(del_items):  # reversed, so that keys remain valid in sequences
            delattr_op(node, key)

        return True, node


class IRNodeDumper(IRNodeMapper):
    @classmethod
    def apply(cls, root_node, *, as_json=False):
        return cls(as_json=as_json)(root_node)

    def __init__(self, as_json: bool):
        self.as_json = as_json

    def __call__(self, node):
        result = self.visit(copy.deepcopy(node))
        if self.as_json:
            result = gt_utils.jsonify(result)
        return result

    def visit_Node(self, path: tuple, node_name: str, node: Node):
        object_name = node.__class__.__name__.split(".")[-1]
        keep_node, new_node = self.generic_visit(path, node_name, node)
        return keep_node, {object_name: new_node.as_dict()}


dump_ir = IRNodeDumper.apply


def iter_nodes_of_type(root_node: Node, node_type: Type) -> Generator[Node, None, None]:
    """Yield an iterator over the nodes of node_type inside root_node in DFS order."""

    def recurse(node: Node) -> Generator[Node, None, None]:
        for key, value in iter_attributes(node):
            if isinstance(node, collections.abc.Iterable):
                if isinstance(node, collections.abc.Mapping):
                    children = node.values()
                else:
                    children = node
            else:
                children = gtc_utils.listify(value)

            for value in children:
                if isinstance(value, Node):
                    yield from recurse(value)

            if isinstance(node, node_type):
                yield node

    yield from recurse(root_node)


def location_to_source_location(loc: Optional[Location]) -> Optional[common.SourceLocation]:
    if loc is None or loc.line <= 0 or loc.column <= 0:
        return None
    return common.SourceLocation(loc.line, loc.column, loc.scope)
