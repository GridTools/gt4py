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
from pydantic import StrictFloat, StrictInt, StrictStr
import sys
import typing
from typing import Union
import typing_inspect

import gtc_unstructured.irs.common
from eve import type_definitions
from eve.utils import UIDGenerator

from . import ast_node_matcher as anm
from . import gtscript_ast
from .ast_node_matcher import Capture, Transformer

class StrToSymbolTransformer(Transformer):
    @staticmethod
    def transform(capture: str):
        return ast.Name(id=capture)

    @staticmethod
    def invert(transformed_capture: ast.Name):
        assert isinstance(transformed_capture, ast.Name)
        return transformed_capture.id

class SubscriptTransformer(Transformer):
    @staticmethod
    def transform(capture: Union[ast.Name, ast.Tuple]):
        if isinstance(capture, ast.Tuple):
            assert len(capture.elts) > 1
            return capture.elts
        return [capture]

    @staticmethod
    def invert(transformed_capture):
        if len(transformed_captures) > 1:
            return ast.Tuple(elts=transformed_capture)
        return transformed_capture


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
            type_definitions.SymbolRef,
            type_definitions.SymbolName,
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

        SymbolName = ast.Name(id=Capture(0))

        SymbolRef = ast.Name(id=Capture("name"))

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

        LocationSpecification = ast.withitem(
            context_expr=ast.Call(
                func=ast.Name(id="location"), args=[Capture("location_type")]
            ),
            optional_vars=Capture(
                "name", default=lambda: ast.Name(id=UIDGenerator.sequential_id(prefix="location"))
            ),
        )

        Subscript = ast.Subscript(
            value=Capture("value"), slice=ast.Index(Capture("indices", transformer=SubscriptTransformer))
        )

        BinaryOp = ast.BinOp(op=Capture("op"), left=Capture("left"), right=Capture("right"))

        Call = ast.Call(args=Capture("args"), func=ast.Name(id=Capture("func")))

        SubscriptCall = ast.Call(args=Capture("args"), func=Capture("func", expected_type=ast.Subscript))

        List_ = ast.List(elts=Capture("elts"))

        LocationComprehension = ast.comprehension(
            target=Capture("target"), iter=Capture("iterable")
        )

        Generator = ast.GeneratorExp(generators=Capture("generators"), elt=Capture("elt"))

        Assign = ast.Assign(targets=[Capture("target")], value=Capture("value"))

        Stencil = ast.With(items=Capture("iteration_spec"), body=Capture("body"))

        Pass = ast.Pass()

        Argument = ast.arg(arg=Capture("name", transformer=StrToSymbolTransformer), annotation=Capture("type_"))

        Computation = ast.FunctionDef(
            #args=ast.arguments(args=Capture("arguments")),
            body=Capture("stencils"),
            name=Capture("name"),
        )

    leaf_map = {
        ast.Mult: gtc_unstructured.irs.common.BinaryOperator.MUL,
        ast.Add: gtc_unstructured.irs.common.BinaryOperator.ADD,
        ast.Sub: gtc_unstructured.irs.common.BinaryOperator.SUB,
        ast.Div: gtc_unstructured.irs.common.BinaryOperator.DIV,
        ast.Pass: gtscript_ast.Pass,
    }

    # Some types appearing in the python ast are mapped to different types in the gtscript_ast. This dictionary
    #  maps from the type appearing in the python ast to potential types in the gtscript ast. Note that this
    #  mapping does not break the 1-to-1 correspondence between python and gtscript ast, as the mapped types
    #  must only appear once for a given field.
    pseudo_polymorphic_types = {
        str: set([type_definitions.SymbolRef, StrictStr]),
        int: set([StrictInt]),
        float: set([StrictFloat])
    }

    # todo(tehrengruber): enhance docstring describing the algorithm
    def transform(self, node, eligible_node_types=None, node_init_args=None):
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

                    if node_init_args:
                        captures = {**node_init_args, **captures}

                    args = list(value for key, value in captures.items() if isinstance(key, int))
                    kwargs = {key: value for key, value in captures.items() if isinstance(key, str)}

                    transformed_kwargs = {}
                    for name, capture in kwargs.items():
                        assert (
                            name in node_type.__annotations__
                        ), f"Invalid capture. No field named `{name}` in `{str(node_type)}`"
                        field_type = node_type.__annotations__[name]
                        if typing_inspect.get_origin(field_type) == list:
                            # determine eligible capture types
                            el_type = typing_inspect.get_args(field_type)[0]
                            eligible_capture_types = self._all_subclasses(el_type, module=module)

                            # transform captures recursively
                            transformed_kwargs[name] = []
                            for child_capture in capture:
                                transformed_kwargs[name].append(
                                    self.transform(child_capture, eligible_capture_types)
                                )
                        else:
                            # determine eligible capture types
                            eligible_capture_types = self._all_subclasses(field_type, module=module)
                            # transform captures recursively
                            transformed_kwargs[name] = self.transform(
                                capture, eligible_capture_types
                            )

                    assert len(args)+len(transformed_kwargs) == len(captures)

                    try:
                        node_type(*args, **transformed_kwargs)
                    except:
                        bla=1+1

                    return node_type(*args, **transformed_kwargs)

                raise ValueError(
                    "Expected a node of type {}".format(
                        ", ".join([ent.__name__ for ent in eligible_node_types])
                    )
                )
        elif type(node) in eligible_node_types:
            return node
        elif type(node) in self.pseudo_polymorphic_types and len(self.pseudo_polymorphic_types[type(node)] & set(eligible_node_types)) > 0:
            valid_types = self.pseudo_polymorphic_types[type(node)] & set(eligible_node_types)
            if len(valid_types) > 1:
                raise RuntimeError(
                    "Invalid gtscript ast specification. The node {node} has multiple valid types in the gtscript ast: {valid_types}")
            return next(iter(valid_types))(node)

        raise ValueError(
            "Expected a node of type {}, but got {}".format(
                {*eligible_node_types, ast.AST}, type(node)
            )
        )
