# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Visitor classes to work with IR trees."""

from __future__ import annotations

import collections.abc
import copy
from typing import ClassVar

from . import concepts, trees
from .extended_typing import Any
from .type_definitions import NOTHING


class NodeVisitor:
    """Simple node visitor class based on :class:`ast.NodeVisitor`.

    A NodeVisitor instance walks a node tree and calls a visitor
    function for every item found. This class is meant to be subclassed,
    with the subclass adding visitor methods.

    Visitor functions for tree elements are named with a standard
    pattern: ``visit_`` + class name of the node. Thus, the visitor
    function for a ``BinOpExpr`` node class should be called ``visit_BinOpExpr``.
    If no visitor function exists for a specific node, the dispatcher
    mechanism looks for the visitor of each one of its parent classes
    in the order define by the class' ``__mro__`` attribute. Finally,
    if no visitor function has been found, the ``generic_visit`` visitor
    is used instead. A concise summary of the steps followed to find an
    appropriate visitor function is:

        1. A ``self.visit_NODE_CLASS_NAME()`` method where `NODE_CLASS_NAME`
           matches ``type(node).__name__``.
        2. A ``self.visit_NODE_BASE_CLASS_NAME()`` method where
           `NODE_BASE_CLASS_NAME` matches ``base.__name__``, and `base` is
           one of the node base classes (evaluated following the order
           given in ``type(node).__mro__``).
        3. ``self.generic_visit()``.

    This dispatching mechanism is implemented in the main :meth:`visit`
    method and can be overridden in subclasses. Therefore, a simple way to extend
    the behavior of a visitor is by inheriting from lightweight `trait` classes
    with a custom ``visit()`` method, which wraps the call to the superclass'
    ``visit()`` and adds extra pre and post visit logic. Check :mod:`eve.traits`
    for further information.

    Note that return values are not forwarded to the caller in the default
    :meth:`generic_visit` implementation. If you want to return a value from
    a nested node in the tree, make sure all the intermediate nodes explicitly
    return children values.

    The recommended idiom to use and define NodeVisitors can be summarized as:

        * Inside visitor functions:

            + call ``self.visit(child_node)`` to visit children.
            + call ``self.generic_visit(node)`` to continue the tree
              traversal in the usual way.

        * Define an ``apply()`` `classmethod` as a shortcut to create an
          instance and start the visit::

                class Visitor(NodeVisitor)
                    @classmethod
                    def apply(cls, tree, init_var, foo, bar=5, **kwargs):
                        instance = cls(init_var)
                        return instance(tree, foo=foo, bar=bar, **kwargs)

                    ...

                result = Visitor.apply(...)

        * If the visitor has internal state, make sure visitor instances
          are never reused or clean up the state at the end.

    Notes:
        If you want to apply changes to nodes during the traversal,
        use the :class:`NodeTranslator` subclass, which handles correctly
        structural modifications of the visited tree.

    """

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        visitor = self.generic_visit

        class_name = node.__class__.__name__
        if "__" in class_name:
            # For concretized data model classes, use the generic name
            class_name = class_name[: class_name.find("__")]
        method_name = "visit_" + class_name

        if hasattr(self, method_name):
            visitor = getattr(self, method_name)
        elif isinstance(node, concepts.Node):
            for node_class in node.__class__.__mro__[1:]:
                class_name = node_class.__name__
                if "__" in class_name:
                    class_name = class_name[: class_name.find("__")]
                method_name = "visit_" + class_name

                if hasattr(self, method_name):
                    visitor = getattr(self, method_name)
                    break

                if node_class is concepts.Node:
                    break

        return visitor(node, **kwargs)

    def generic_visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        for child in trees.iter_children_values(node):
            self.visit(child, **kwargs)

        return None


class NodeTranslator(NodeVisitor):
    """Special `NodeVisitor` to translate nodes and trees.

    A NodeTranslator instance will walk the tree exactly as a regular
    :class:`NodeVisitor` while building an output tree using the return
    values of the visitor methods. If the return value is :obj:`eve.NOTHING`,
    the node will be removed from its location in the output tree,
    otherwise it will be replaced with this new value. The default visitor
    method (:meth:`generic_visit`) returns a `deepcopy` of the original
    node.

    Keep in mind that if the node you're operating on has child nodes
    you must either transform the child nodes yourself or call the
    :meth:`generic_visit` method for the node first.

    Usually you use a NodeTranslator like this::

       output_node = YourTranslator.apply(input_node)

    Notes:
        Check :class:`NodeVisitor` documentation for more details.

    """

    PRESERVED_ANNEX_ATTRS: ClassVar[tuple[str, ...]] = ()

    def generic_visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        memo = kwargs.get("__memo__", None)

        if isinstance(node, concepts.Node):
            new_node = node.__class__(  # type: ignore
                **{
                    name: new_child
                    for name, child in node.iter_children_items()
                    if (new_child := self.visit(child, **kwargs)) is not NOTHING
                }
            )
            if self.PRESERVED_ANNEX_ATTRS and (old_annex := getattr(node, "__node_annex__", None)):
                # access to `new_node.annex` implicitly creates the `__node_annex__` attribute in the property getter
                new_annex_dict = new_node.annex.__dict__
                for key in self.PRESERVED_ANNEX_ATTRS:
                    if (value := getattr(old_annex, key, NOTHING)) is not NOTHING:
                        assert key not in new_annex_dict
                        new_annex_dict[key] = value

            return new_node

        if isinstance(node, (list, tuple, set, collections.abc.Set)) or (
            isinstance(node, collections.abc.Sequence) and not isinstance(node, (str, bytes))
        ):
            # Sequence or set: create a new container instance with the new values
            return node.__class__(  # type: ignore
                new_child
                for child in trees.iter_children_values(node)
                if (new_child := self.visit(child, **kwargs)) is not NOTHING
            )

        if isinstance(node, (dict, collections.abc.Mapping)):
            # Mapping: create a new mapping instance with the new values
            return node.__class__(  # type: ignore[call-arg]
                {
                    name: new_child
                    for name, child in trees.iter_children_items(node)
                    if (new_child := self.visit(child, **kwargs)) is not NOTHING
                }
            )

        return copy.deepcopy(node, memo=memo)
