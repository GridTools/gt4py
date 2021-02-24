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
from .gtscript import Field, Location, Connectivity, TemporaryField, SparseField, TemporarySparseField, LocalField
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
from .passes.const_expr_evaluator import evaluate_const_expr
from .passes.type_deduction import deduce_type

_reduction_mapping = {
    "sum": gtir.ReduceOperator.ADD,
    "product": gtir.ReduceOperator.MUL,
    "min": gtir.ReduceOperator.MIN,
    "max": gtir.ReduceOperator.MAX,
}


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
        loc_type = evaluate_const_expr(symtable, node.location_type)

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
        # resolve call
        func = symtable[node.func.name].value
        method = func.find(*(deduce_type(symtable, arg) for arg in node.args))

        #if node.func in built_in_functions.native_functions and all(issubclass(arg_type, (numbers.Number, Field)) for arg_type in arg_types):
        if method in built_in_functions._native_functions:
            assert len(node.args) == common.NativeFunction.IR_OP_TO_NUM_ARGS[method.name]

            native_func_call = gtir.NativeFuncCall(
                func=common.NativeFunction(method.name),
                args=self.visit(node.args, location_stack=location_stack, symtable=symtable, **kwargs),
                location_type=location_stack[-1][1]
            )
            return native_func_call

        elif method in built_in_functions._neighbor_reductions:
            assert len(node.args) == 1
            if not isinstance(node.args[0], Generator) or len(node.args[0].generators) != 1:
                raise ValueError("Invalid argument to {node.func}")

            op = _reduction_mapping[method.name]
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
        func = evaluate_const_expr(symtable, node.func)

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

            index_types = tuple(deduce_type(symtable, idx) for idx in node.indices)
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
        right_type = deduce_type(symtable, node.value)
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
                location_name = node.value.generators[0].target

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
                # TODO: from Field[] to avoid breaking the TypeDeduction mechanism
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
