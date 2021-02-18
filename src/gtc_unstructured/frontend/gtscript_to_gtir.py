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
import copy
from typing import Any, Dict, List, Union, cast
import numbers

from devtools import debug

import eve
from eve.utils import UIDGenerator
import gtc_unstructured.irs.common as common
import gtc_unstructured.irs.gtir as gtir
from gtc import common as stable_gtc_common

from . import built_in_functions
from .built_in_types import BuiltInTypeMeta, BuiltInType
from .gtscript import Field, Location, Connectivity, TemporaryField, SparseField, TemporarySparseField
from .gtscript_ast import (
    Argument,
    Assign,
    BinaryOp,
    Call,
    Computation,
    Constant,
    Generator,
    Interval,
    IterationOrder,
    LocationComprehension,
    LocationSpecification,
    Pass,
    Stencil,
    Subscript,
    SubscriptCall,
    SymbolRef,
    SymbolName,
    External,
    TemporaryFieldDecl,
    TemporarySparseFieldDecl,
    UnaryOp
)

_reduction_mapping = {
    "sum": gtir.ReduceOperator.ADD,
    "product": gtir.ReduceOperator.MUL,
    "min": gtir.ReduceOperator.MIN,
    "max": gtir.ReduceOperator.MAX,
}


class SymbolTable:
    """
    A simple symbol table containing all the types of all symbols and potentially their values if known at compile
    time
    """

    def __init__(self, types: Dict[str, Any], constants: Dict[str, Any]):
        # Currently supported types: BuiltInType, LocationType, DataType
        self.types = types
        self.constants = constants

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.types

    def __getitem__(self, symbol: str) -> Any:
        return self.types[symbol]

    def __setitem__(self, symbol: str, val: Any):
        if symbol in self.types:
            if (
                    self.types[symbol] == val
            ):  # TODO(tehrengruber): just a workaround. remove when symbol table has proper scope!
                return self.types[symbol]
            raise ValueError(f"Symbol `{symbol}` already in symbol table.")

        self.types[symbol] = val
        return self.types[symbol]

    # todo(tehrengruber): decide on constant folding: what, when, how?
    def materialize_constant(self, symbol: str, expected_type=None):
        """
        Materialize constant `symbol`, i.e. return the value of that symbol.

        Currently the only constants are types, but this is currently the place where constant folding would happen,
        hence the name.

        Example:
        .. code-block:: python

            self._materialize_constant("Vertex") == LocationType.Vertex
        """
        if symbol not in self.types:
            raise ValueError(f"Symbol {symbol} not found")
        if symbol not in self.constants:
            raise ValueError(f"Symbol {symbol} : {self.types[symbol]} is not a constant")
        val = self.constants[symbol]
        if expected_type is not None and not isinstance(val, expected_type):
            raise ValueError(
                f"Expected a symbol {symbol} of type {expected_type}, but got {self.types[symbol]}"
            )
        return val


class NodeCanonicalizer(eve.NodeTranslator):
    @classmethod
    def apply(cls, gt4py_ast: Computation):
        return cls().visit(gt4py_ast)

    def visit_Computation(self, node: Computation):
        # canonicalize nested stencils
        stencils = []
        for stencil in node.stencils:
            if all(isinstance(body_node, Stencil) for body_node in stencil.body):
                # if we find a nested stencil flatten it
                for nested_stencil in stencil.body:
                    assert isinstance(nested_stencil, Stencil)
                    # TODO(tehrengruber): validate iteration_spec otherwise the TemporaryFieldDeclExtractor fails
                    flattened_stencil = Stencil(
                        iteration_spec=self.generic_visit(stencil.iteration_spec)
                                       + self.generic_visit(nested_stencil.iteration_spec),
                        body=self.generic_visit(nested_stencil.body),
                    )
                    if any(isinstance(body_node, Stencil) for body_node in flattened_stencil.body):
                        raise ValueError("Nesting a stencil inside a nested stencil not allowed.")
                    stencils.append(flattened_stencil)
            elif not any(isinstance(body_node, Stencil) for body_node in stencil.body):
                # if we have a non-nested stencil just keep it as is
                stencils.append(self.visit(stencil))
            else:
                raise ValueError("Mixing nested and unnested stencils not allowed.")

        # write back changes to node
        node.stencils = stencils

        return node

    def visit_UnaryOp(self, node: UnaryOp):
        if isinstance(node.operand, Constant):
            if node.op == stable_gtc_common.UnaryOperator.NEG:
                return Constant(value=-node.operand.value)
            elif node.op == stable_gtc_common.UnaryOperator.POS:
                return node.operand

        raise NotImplementedError()

    def visit_Call(self, node: Call):
        # TODO(tehrengruber): this could be done by the call inliner
        # neighbor accessor canonicalization
        # neighbor_selector_mapping = {
        #    "vertices": Symbol(name="Vertex"),  # TODO(tehrengruber): common.LocationType.Vertex,
        #    "edges": Symbol(name="Edge"),  # common.LocationType.Edge,
        #    "cells": Symbol(name="Cell"),  # common. LocationType.Cell
        # }
        # if node.func in neighbor_selector_mapping:
        #    return Call(
        #        func="neighbors",
        #        args=[self.generic_visit(node.args[0]), neighbor_selector_mapping[node.func]],
        #    )

        return self.generic_visit(node)


class ConstExprEvaluator(eve.NodeVisitor):
    @classmethod
    def apply(cls, symtable, values):
        instance = cls()
        return instance.visit(values, symtable=symtable)

    def visit_Node(self, node, **kwargs):
        raise ValueError("Evaluation failed")

    def visit_SymbolRef(self, node: SymbolRef, *, symtable, **kwargs):
        return self.visit(symtable[node.name])

    def visit_External(self, node: External, **kwargs):
        return node.value

    def visit_Constant(self, node: Constant, **kwargs):
        return node.value

    def visit_Subscript(self, node: Subscript, **kwargs):
        return self.visit(node.value, **kwargs)[tuple(self.visit(idx, **kwargs) for idx in node.indices)]


class RewriteSparseFieldAccess(eve.NodeTranslator):
    @classmethod
    def apply(cls, from_, to, values):
        instance = cls()
        return instance.visit(values, from_=from_, to=to)

    def visit_FieldAccess(self, node: gtir.FieldAccess, *, from_, to):
        if len(node.subscript) == 2 and node.subscript[0].name == from_[0] and node.subscript[1].name == from_[1]:
            return gtir.FieldAccess(name=node.name, subscript=[gtir.LocationRef(name=to)], location_type=node.location_type)
        return node


from .built_in_types import LocalField


class TypeInference(eve.NodeVisitor):
    @classmethod
    def apply(cls, symtable, values):
        instance = cls()
        return instance.visit(values, symtable=symtable)

    def visit_Node(self, node, *, symtable, **kwargs):
        raise ValueError(f"Type of node {node} not defined.")

    def visit_Argument(self, node: Argument, **kwargs):
        return node.type_

    def visit_TemporaryFieldDecl(self, node: TemporaryFieldDecl, **kwargs):
        return node.type_

    def visit_TemporarySparseFieldDecl(self, node: TemporarySparseFieldDecl, **kwargs):
        return node.type_

    def visit_Call(self, node: Call, **kwargs):
        # todo: enhance
        return numbers.Number
        # return built_in_functions[node.func].return_type(*self.visit(node.args))

    def visit_Constant(self, node: Constant, **kwargs):
        return type(node.value)

    def visit_SymbolRef(self, node: SymbolRef, symtable, **kwargs):
        return self.visit(symtable[node.name], symtable=symtable, **kwargs)

    def visit_External(self, node: External, **kwargs):
        return type(node.value)

    def visit_LocationSpecification(self, node: LocationSpecification, *, symtable, **kwargs):
        return Location[ConstExprEvaluator.apply(symtable, node.location_type)]

    def visit_LocationComprehension(self, node: LocationComprehension, *, symtable, **kwargs):
        connectivity = symtable[node.iterable.value.name].type_
        index_type = self.visit(node.iterable.indices[0], symtable=symtable, **kwargs)
        if index_type != Location[connectivity.primary_location()]:
            raise ValueError(
                f"You are trying to access a connectivity posed on {connectivity.primary_location()} using a location of type {index_type}")
        return Location[connectivity.secondary_location()]

    def visit_SubscriptCall(self, node: SubscriptCall, *, symtable, **kwargs):
        func = ConstExprEvaluator.apply(symtable, node.func)
        if issubclass(func, BuiltInType):
            return func
        raise ValueError(f"Type of node {node} not defined.")

    def visit_BinaryOp(self, node: BinaryOp, **kwargs):
        # todo: enhance
        return numbers.Number

    def visit_Subscript(self, node: Subscript, *, symtable, **kwargs):
        # todo: enhance
        if all(isinstance(symtable[idx.name], (LocationSpecification, LocationComprehension))
               for idx in node.indices):
            # todo: use Number
            return numbers.Number
        raise ValueError(f"Type of node {node} not defined.")

    def visit_Generator(self, node: LocationComprehension, *, symtable, **kwargs):
        connectivity = symtable[node.generators[0].iterable.value.name].type_
        assert issubclass(connectivity, Connectivity)
        return LocalField[connectivity]


class GTScriptToGTIR(eve.NodeTranslator):
    # TODO(tehrengruber): the current way of passing the location_stack is tidious and error prone

    @classmethod
    def apply(cls, gt4py_ast: Computation):
        return cls().visit(gt4py_ast)

    def visit_IterationOrder(self, node: IterationOrder, **kwargs) -> gtir.common.LoopOrder:
        return gtir.common.LoopOrder[node.order]

    def visit_Interval(self, node: Interval, **kwargs):
        return None

    def visit_LocationSpecification(
            self, node: LocationSpecification, *, symtable, **kwargs
    ) -> gtir.LocationComprehension:
        loc_type = ConstExprEvaluator.apply(symtable, node.location_type)

        return gtir.PrimaryLocation(name=node.name, location_type=loc_type)

    def visit_LocationComprehension(
            self, node: LocationComprehension, *, symtable, location_stack, **kwargs
    ) -> gtir.LocationComprehension:
        assert isinstance(symtable[node.iterable.value.name], Argument)
        assert issubclass(symtable[node.iterable.value.name].type_, Connectivity)
        connectivity_name = node.iterable.value.name
        connectivity = symtable[connectivity_name].type_.base_connectivity()
        if not issubclass(connectivity, Connectivity):
            # todo(tehrengruber): better error message
            raise ValueError(
                f"Invalid neighbor specification. Expected a connectivity access."
            )

        of = self.visit(node.iterable.value, symtable=symtable, location_stack=location_stack, **kwargs)
        if not isinstance(of, gtir.SymbolRef):
            raise ValueError(f"Expected a `SymbolRef` node, but got `{type(of)}`")

        return gtir.LocationComprehension(name=node.target, of=gtir.ConnectivityRef(name=of))

    def visit_Call(self, node: Call, *, location_stack, symtable, **kwargs):
        # TODO(tehrengruber): all of this can be done with the symbol table and the call inliner
        # reductions

        args_contain_generator = any(isinstance(arg, Generator) for arg in node.args)
        arg_types = list(TypeInference.apply(symtable, arg) for arg in node.args)

        # method = built_in_functions.native_functions[node.func]
        # TODO(workshop): would be nice if we could use `method.applicable`

        if node.func in built_in_functions.native_functions and all(issubclass(arg_type, (numbers.Number, Field)) for arg_type in arg_types):
            if len(node.args) != gtir.NativeFunction.IR_OP_TO_NUM_ARGS[node.func]:
                raise ValueError()

            native_func_call = gtir.NativeFuncCall(
                func=gtir.NativeFunction(node.func),
                args=self.visit(node.args, location_stack=location_stack, symtable=symtable, **kwargs),
                location_type=location_stack[-1][1]
            )
            return native_func_call

        elif node.func in built_in_functions.neighbor_reductions:
            if any(method.applicable(arg_types) for method in getattr(built_in_functions, node.func).methods):
                if not len(node.args):
                    raise ValueError(
                        f"Invalid number of arguments specified for function {node.func}. Expected 1, but {len(node.args)} were given."
                    )
                if not isinstance(node.args[0], Generator) or len(node.args[0].generators) != 1:
                    raise ValueError("Invalid argument to {node.func}")

                op = _reduction_mapping[node.func]
                neighbors = self.visit(
                    node.args[0].generators[0], **{**kwargs, "symtable": symtable, "location_stack": location_stack}
                )

                # operand gets new location stack
                new_location_stack = location_stack + [(node.args[0].generators[0].target, symtable[node.args[0].generators[0].iterable.value.name].type_.secondary_location())]

                operand = self.visit(
                    node.args[0].elt, **{**kwargs, "symtable": {**symtable, **node.args[0].symtable_}, "location_stack": new_location_stack}
                )

                return gtir.NeighborReduce(
                    op=op,
                    operand=operand,
                    neighbors=neighbors,
                    location_type=location_stack[-1][1],
                )

        raise ValueError(
            "Could not resolve call to function `{node.func}. Either the function does not exist or the arguments don't match.`")

    def visit_SubscriptCall(self, node: SubscriptCall, symtable, location_stack, inside_sparse_assign,
                            neighbor_vector_access_expr_location_name, **kwargs):
        assert inside_sparse_assign
        func = ConstExprEvaluator.apply(symtable, node.func)

        return gtir.NeighborVectorAccess(
            exprs=self.visit(node.args[0].elts, **{**kwargs, "symtable": symtable, "location_stack": location_stack}),
            location_ref=gtir.LocationRef(name=neighbor_vector_access_expr_location_name),
            location_type=location_stack[0][1],  # wrong
        )

    def visit_Constant(self, node: Constant, *, location_stack, **kwargs):
        py_dtype_to_gtir = {  # TODO(tehrengruber): check
            int: common.DataType.INT32,
            float: common.DataType.FLOAT64,
        }
        return gtir.Literal(
            value=str(node.value),
            vtype=next(gtir_type for type_, gtir_type in py_dtype_to_gtir.items() if isinstance(node.value, type_)),
            location_type=location_stack[-1][1],
        )

    def visit_SymbolRef(self, node: SymbolRef, *, symtable, location_stack, **kwargs):
        assert node.name in symtable
        # todo: Argument and TemporaryFieldDecl should have same base class
        # TODO: SparseField's should have slice syntax
        if isinstance(symtable[node.name], (Argument, TemporaryFieldDecl, TemporarySparseFieldDecl)):
            if issubclass(symtable[node.name].type_, (Field, SparseField, TemporaryField, TemporarySparseField)):
                return gtir.FieldAccess(
                    name=node.name,
                    location_type=location_stack[-1][1],
                    subscript=[gtir.LocationRef(name=location_stack[0][0])],
                )  # TODO(tehrengruber): just visit the subscript symbol
            elif issubclass(symtable[node.name].type_, Connectivity):
                return gtir.SymbolRef(node.name)

        raise ValueError()

    def visit_Subscript(self, node: Subscript, *, symtable, location_stack, **kwargs):
        value_decl = symtable[node.value.name]
        if isinstance(value_decl, (Argument, TemporaryFieldDecl, TemporarySparseFieldDecl)) and (
                issubclass(value_decl.type_, (Field, TemporaryField, SparseField, TemporarySparseField))):
            try:
                if not all(isinstance(symtable[index.name], LocationSpecification) or isinstance(symtable[index.name],
                                                                                                 LocationComprehension) for
                           index in node.indices):
                    1+1
            except:
                bla=1+1
            assert len(node.indices) in [1, 2]
            assert all(isinstance(index, SymbolRef) for index in node.indices)
            assert all(isinstance(symtable[index.name], LocationSpecification) or isinstance(symtable[index.name],
                                                                                             LocationComprehension) for
                       index in node.indices)

            # check arguments for consistency
            # TODO: lower IRs should check this too, currently without this check they just generate invalid code
            if issubclass(value_decl.type_, SparseField) or issubclass(value_decl.type_, TemporarySparseField):
                connectivity = value_decl.type_.args[0]
                assert issubclass(connectivity, Connectivity)
                expected_index_types = (
                    Location[connectivity.primary_location()], Location[connectivity.secondary_location()])
            elif issubclass(value_decl.type_, Field) or issubclass(value_decl.type_, TemporaryField):
                expected_index_types = (Location[value_decl.type_.args[0]],)
            else:
                raise RuntimeError(
                    f"Invalid symbol '{node.value.name}' in subscript expression ({node})"
                )

            index_types = tuple(TypeInference.apply(symtable, idx) for idx in node.indices)
            assert index_types == expected_index_types

            # TODO(tehrengruber): just visit the index symbol
            return gtir.FieldAccess(
                name=gtir.SymbolRef(node.value.name),
                subscript=[
                    gtir.LocationRef(name=index.name) for index in cast(List[SymbolRef], node.indices)
                ],
                location_type=location_stack[-1][1],
            )

        raise ValueError()

    def visit_Generator(self, node: Generator, *, symtable, **kwargs):
        return self.generic_visit(node, symtable={**symtable, **node.symtable_}, **kwargs)

    def visit_Assign(self, node: Assign, *, symtable, location_stack, **kwargs) -> Union[
        gtir.AssignStmt, gtir.NeighborAssignStmt]:
        right_type = TypeInference.apply(symtable, node.value)
        sparse_assign = issubclass(right_type, LocalField)
        if sparse_assign:
            location_type = symtable[node.target.name].type_.args[0]
            if location_type != right_type.args[0]:
                raise ValueError()
            # gtir expects the name of the connectivity instead of its type, hence search for the name
            connectivity_name = None
            for symbol in symtable.values():
                if isinstance(symbol, Argument) and issubclass(
                        symbol.type_, Connectivity) and location_type == symbol.type_:
                    assert connectivity_name is None
                    connectivity_name = symbol.name
            assert connectivity_name is not None
            location_name = UIDGenerator.sequential_id(prefix="location")

            if isinstance(node.value, Generator):
                # TODO(tehrengruber) hacky, visit the generator
                new_location_stack = location_stack + [(node.value.generators[0].target, symtable[
                    node.value.generators[0].iterable.value.name].type_.secondary_location())]
                assert node.value.generators[0].iterable.value.name == connectivity_name
                right = self.visit(node.value.elt, symtable={**symtable, **node.value.symtable_}, location_stack=location_stack, **kwargs)
                right = RewriteSparseFieldAccess.apply((location_stack[0][0], node.value.generators[0].target), location_name, right)

            else:
                right = self.visit(node.value, symtable=symtable,
                                 location_stack=location_stack,
                                 inside_sparse_assign=sparse_assign,
                                 neighbor_vector_access_expr_location_name=location_name, **kwargs)

            return gtir.NeighborAssignStmt(
                left=self.visit(node.target, symtable=symtable, location_stack=location_stack, **kwargs),
                right=right,
                location_type=location_stack[-1][1],
                neighbors=gtir.LocationComprehension(name=location_name,
                                                     of=gtir.ConnectivityRef(name=connectivity_name))
            )
        else:
            return gtir.AssignStmt(
                left=self.visit(node.target, **{"symtable": symtable, "location_stack": location_stack, **kwargs}),
                right=self.visit(node.value, **{"symtable": symtable, "location_stack": location_stack,
                                                "inside_sparse_assign": sparse_assign, **kwargs}),
                location_type=location_stack[-1][1],
            )

    def visit_BinaryOp(self, node: BinaryOp, *, symtable, location_stack, **kwargs):
        return gtir.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **{"symtable": symtable, "location_stack": location_stack, **kwargs}),
            right=self.visit(node.right, **{"symtable": symtable, "location_stack": location_stack, **kwargs}),
            location_type=location_stack[-1][1],
        )

    def visit_Stencil(self, node: Stencil, **kwargs) -> gtir.Stencil:
        loop_order, primary_location = None, None
        for it_spec in node.iteration_spec:
            if isinstance(it_spec, IterationOrder):
                assert loop_order is None
                loop_order = self.visit(it_spec, **kwargs)
            elif isinstance(it_spec, LocationSpecification):
                assert primary_location is None
                primary_location = self.visit(it_spec, **kwargs)
            elif isinstance(it_spec, Interval):
                # TODO(tehrengruber): implement
                pass
            else:
                raise ValueError()
        assert loop_order is not None
        assert primary_location is not None

        location_stack = [(primary_location.name,
                           primary_location.location_type)]  # TODO(tehrengruber): we should store dimensions here

        horizontal_loops = []
        for stmt in node.body:
            horizontal_loops.append(
                gtir.HorizontalLoop(
                    stmt=self.visit(stmt, location_stack=location_stack, **kwargs),
                    location=primary_location,
                )
            )

        return gtir.Stencil(
            vertical_loops=[
                gtir.VerticalLoop(loop_order=loop_order, horizontal_loops=horizontal_loops)
            ]
        )

    def visit_Computation(self, node: Computation) -> gtir.Computation:
        field_args = []
        connectivity_args = []
        temporary_fields = []

        for arg in node.arguments + node.declarations:
            if issubclass(arg.type_, Field) or issubclass(arg.type_, TemporaryField):
                field_type = arg.type_
                loc_type, vtype = field_type.args
                if isinstance(loc_type, (common.LocationType, common.VerticalLocationType)):
                    loc_type = [loc_type]

                assert isinstance(vtype, common.DataType)
                assert isinstance(loc_type, list) and all(
                    isinstance(l, (common.LocationType, common.VerticalLocationType)) for l in loc_type
                )

                horizontal = None
                vertical = None
                new_type_args = [None, vtype]
                for loc in loc_type:
                    if isinstance(loc, common.LocationType):
                        horizontal = gtir.HorizontalDimension(primary=loc)
                        new_type_args[0] = loc
                    elif isinstance(loc, common.VerticalLocationType):
                        vertical = gtir.VerticalDimension()
                    else:
                        raise ValueError(f"Invalid field dimension definition '{loc}")

                dims = gtir.Dimensions(horizontal=horizontal, vertical=vertical)

                if issubclass(arg.type_, Field):
                    field_args.append(
                        gtir.UField(name=arg.name, vtype=field_type.args[1], dimensions=dims)
                    )
                elif issubclass(arg.type_, TemporaryField):
                    temporary_fields.append(
                        gtir.TemporaryField(
                            name=arg.name, vtype=field_type.args[1], dimensions=dims
                        )
                    )

                # TODO(hackathon): this hack is removing the VerticalLocationType args
                # TODO: from Field[] to avoid breaking the TypeInference mechanism
                arg.type_.args = tuple(new_type_args)

            elif issubclass(arg.type_, Connectivity):
                base_connectivty = arg.type_.base_connectivity()
                connectivity_args.append(
                    gtir.Connectivity(
                        name=arg.name,
                        primary=base_connectivty.primary_location(),
                        secondary=base_connectivty.secondary_location(),
                        max_neighbors=base_connectivty.max_neighbors(),
                        has_skip_values=base_connectivty.has_skip_values(),
                    )
                )
            elif issubclass(arg.type_, SparseField) or issubclass(arg.type_, TemporarySparseField):
                connectivity = arg.type_.args[0]
                if not isinstance(connectivity, list):
                    connectivity = [connectivity]

                horizontal = None
                vertical = None
                new_type_args = [None, arg.type_.args[1]]
                for c in connectivity:
                    if issubclass(c, Connectivity):
                        horizontal = gtir.HorizontalDimension(
                            primary=c.base_connectivity().primary_location()
                        )
                        new_type_args[0] = c                        
                    elif isinstance(c, common.VerticalLocationType):
                        vertical = gtir.VerticalDimension()
                    else:
                        raise ValueError(f"Invalid field dimension definition '{loc}")

                # TODO(hackathon): this hack is removing the VerticalLocationType args
                # TODO: from Field[] to avoid breaking the TypeInference mechanism
                arg.type_.args = tuple(new_type_args)

                # gtir expects the name of the connectivity instead of its type, hence search for the name
                connectivity_name = None
                for other_arg in node.arguments:
                    if (
                        issubclass(other_arg.type_, Connectivity)
                        and arg.type_.args[0] == other_arg.type_
                    ):
                        assert connectivity_name is None
                        connectivity_name = other_arg.name

                # TODO(tehrengruber): check dims argument with hannes
                dims = gtir.Dimensions(horizontal=horizontal, vertical=vertical)

                if issubclass(arg.type_, SparseField):
                    field_args.append(
                        gtir.SparseField(
                            name=arg.name,
                            connectivity=gtir.SymbolRef(connectivity_name),
                            vtype=arg.type_.args[1],
                            dimensions=dims,
                        )
                    )
                elif issubclass(arg.type_, TemporarySparseField):
                    temporary_fields.append(
                        gtir.TemporarySparseField(
                            name=arg.name,
                            connectivity=gtir.SymbolRef(connectivity_name),
                            vtype=arg.type_.args[1],
                            dimensions=dims,
                        )
                    )

        return gtir.Computation(
            name=node.name,
            connectivities=connectivity_args,
            params=field_args,
            declarations=temporary_fields,
            stencils=self.visit(node.stencils, symtable=node.symtable_),
        )

    def visit_Pass(self, node: Pass):
        raise RuntimeError("Not implemented.")
