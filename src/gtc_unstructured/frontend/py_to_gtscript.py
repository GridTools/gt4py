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
import ast
import enum
import inspect
import sys
import typing

import typing_inspect

import gtc_unstructured.irs.common
from eve import type_definitions
from eve.utils import UIDGenerator

from . import ast_node_matcher as anm
from . import gtscript_ast
from .ast_node_matcher import Capture


class PyToGTScript:
    @staticmethod
    def _all_subclasses(typ, *, module=None):
        """
        Return all subclasses of a given type.

        The type must be one of

         - :class:`GTScriptAstNode` (returns all subclasses of the given class)
         - :class:`Union` (return the subclasses of the united)
         - :class:`ForwardRef` (resolve the reference given the specified module and return its subclasses)
         - built-in python type: :class:`str`, :class:`int`, `type(None)` (return as is)
        """
        if inspect.isclass(typ) and issubclass(typ, gtscript_ast.GTScriptASTNode):
            result = {
                typ,
                *typ.__subclasses__(),
                *[
                    s
                    for c in typ.__subclasses__()
                    for s in PyToGTScript._all_subclasses(c)
                    if not inspect.isabstract(c)
                ],
            }
            return result
        elif inspect.isclass(typ) and typ in [
            gtc_unstructured.irs.common.AssignmentKind,
            gtc_unstructured.irs.common.UnaryOperator,
            gtc_unstructured.irs.common.BinaryOperator,
        ]:
            # note: other types in gtc_unstructured.irs.common, e.g. gtc_unstructured.irs.common.DataType are not valid leaf nodes here as they
            #  map to symbols in the gtscript ast and are resolved there
            assert issubclass(typ, enum.Enum)
            return {typ}
        elif typing_inspect.is_union_type(typ):
            return {
                sub_cls
                for el_cls in typing_inspect.get_args(typ)
                for sub_cls in PyToGTScript._all_subclasses(el_cls, module=module)
            }
        elif isinstance(typ, typing.ForwardRef):
            type_name = typing_inspect.get_forward_arg(typ)
            if not hasattr(module, type_name):
                raise ValueError(
                    f"Reference to type `{type_name}` in `ForwardRef` not found in module {module.__name__}"
                )
            return PyToGTScript._all_subclasses(getattr(module, type_name), module=module)
        elif typ in [
            type_definitions.Str,
            type_definitions.Int,
            type_definitions.Float,
            str,
            int,
            float,
            type(None),
        ]:  # TODO(tehrengruber): enhance
            return {typ}

        raise ValueError(f"Invalid field type {typ}")

    class Patterns:
        """
        Stores the pattern nodes / templates to be used extracting information from the Python ast.

        Patterns are a 1-to-1 mapping from context and Python ast node to GTScript ast node. Context is encoded in the
        field types and all understood sementic is encoded in the structure.
        """

        Symbol = ast.Name(id=Capture("name"))

        IterationOrder = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="computation"), args=[ast.Name(id=Capture("order"))]
            )
        )

        Constant = ast.Constant(value=Capture("value"))

        Interval = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="interval"), args=[Capture("start"), Capture("stop")]
            )
        )

        # TODO(tehrengruber): this needs to be a function, since the uid must be generated each time
        LocationSpecification = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="location"), args=[ast.Name(id=Capture("location_type"))]
            ),
            optional_vars=Capture(
                "name", default=ast.Name(id=UIDGenerator.sequential_id(prefix="location"))
            ),
        )

        SubscriptSingle = ast.Subscript(
            value=Capture("value"), slice=ast.Index(value=ast.Name(id=Capture("index")))
        )

        SubscriptMultiple = ast.Subscript(
            value=Capture("value"), slice=ast.Index(value=ast.Tuple(elts=Capture("indices")))
        )

        BinaryOp = ast.BinOp(op=Capture("op"), left=Capture("left"), right=Capture("right"))

        Call = ast.Call(args=Capture("args"), func=ast.Name(id=Capture("func")))

        LocationComprehension = ast.comprehension(
            target=Capture("target"), iter=Capture("iterator")
        )

        Generator = ast.GeneratorExp(generators=Capture("generators"), elt=Capture("elt"))

        Assign = ast.Assign(targets=[Capture("target")], value=Capture("value"))

        Stencil = ast.With(items=Capture("iteration_spec"), body=Capture("body"))

        Pass = ast.Pass()

        Argument = ast.arg(arg=Capture("name"), annotation=Capture("type_"))

        Computation = ast.FunctionDef(
            args=ast.arguments(args=Capture("arguments")),
            body=Capture("stencils"),
            name=Capture("name"),
        )

    leaf_map = {
        ast.Mult: gtc_unstructured.irs.common.BinaryOperator.MUL,
        ast.Add: gtc_unstructured.irs.common.BinaryOperator.ADD,
        ast.Div: gtc_unstructured.irs.common.BinaryOperator.DIV,
        ast.Pass: gtscript_ast.Pass,
    }

    # todo(tehrengruber): enhance docstring describing the algorithm
    def transform(self, node, eligible_node_types=None):
        """Transform python ast into GTScript ast recursively."""
        if eligible_node_types is None:
            eligible_node_types = [gtscript_ast.Computation]

        if isinstance(node, ast.AST):
            is_leaf_node = len(list(ast.iter_fields(node))) == 0
            if is_leaf_node:
                if not type(node) in self.leaf_map:
                    raise ValueError(
                        f"Leaf node of type {type(node)}, found in the python ast, can not be mapped."
                    )
                return self.leaf_map[type(node)]
            else:
                # visit node fields and transform
                # TODO(tehrengruber): check if multiple nodes match and throw an error in that case
                # disadvantage: templates can be ambiguous
                for node_type in eligible_node_types:
                    if not hasattr(self.Patterns, node_type.__name__):
                        continue
                    captures = {}
                    if not anm.match(
                        node, getattr(self.Patterns, node_type.__name__), captures=captures
                    ):
                        continue
                    module = sys.modules[node_type.__module__]
                    transformed_captures = {}
                    for name, capture in captures.items():
                        assert (
                            name in node_type.__annotations__
                        ), f"Invalid capture. No field named `{name}` in `{str(node_type)}`"
                        field_type = node_type.__annotations__[name]
                        if typing_inspect.get_origin(field_type) == list:
                            # determine eligible capture types
                            el_type = typing_inspect.get_args(field_type)[0]
                            eligible_capture_types = self._all_subclasses(el_type, module=module)

                            # transform captures recursively
                            transformed_captures[name] = []
                            for child_capture in capture:
                                transformed_captures[name].append(
                                    self.transform(child_capture, eligible_capture_types)
                                )
                        else:
                            # determine eligible capture types
                            eligible_capture_types = self._all_subclasses(field_type, module=module)
                            # transform captures recursively
                            transformed_captures[name] = self.transform(
                                capture, eligible_capture_types
                            )
                    return node_type(**transformed_captures)
                raise ValueError(
                    "Expected a node of type {}".format(
                        ", ".join([ent.__name__ for ent in eligible_node_types])
                    )
                )
        elif type(node) in eligible_node_types:
            return node

        raise ValueError(
            "Expected a node of type {}, but got {}".format(
                {*eligible_node_types, ast.AST}, type(node)
            )
        )
