# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
import dataclasses
import functools
import inspect

from gt4py import eve
from gt4py.eve import concepts
from gt4py.eve.extended_typing import Any, Callable, Optional, TypeVar, Union
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_call_to
from gt4py.next.iterator.type_system import rules, type_specifications as it_ts
from gt4py.next.type_system import type_info, type_specifications as ts


def _is_representable_as_int(s: int | str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_compatible_type(type_a: ts.TypeSpec, type_b: ts.TypeSpec):
    """
    Predicate to determine if two types are compatible.

    This function gracefully handles iterators with unknown positions which are considered
    compatible to any other positions of another iterator. Beside that this function
    simply checks for equality of types.

    >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    >>> IDim = common.Dimension(value="IDim")
    >>> type_on_i_of_i_it = it_ts.IteratorType(
    ...     position_dims=[IDim], defined_dims=[IDim], element_type=bool_type
    ... )
    >>> type_on_undefined_of_i_it = it_ts.IteratorType(
    ...     position_dims="unknown", defined_dims=[IDim], element_type=bool_type
    ... )
    >>> _is_compatible_type(type_on_i_of_i_it, type_on_undefined_of_i_it)
    True

    >>> JDim = common.Dimension(value="JDim")
    >>> type_on_j_of_j_it = it_ts.IteratorType(
    ...     position_dims=[JDim], defined_dims=[JDim], element_type=bool_type
    ... )
    >>> _is_compatible_type(type_on_i_of_i_it, type_on_j_of_j_it)
    False
    """
    is_compatible = True

    def is_compatible_element(el_type_a: ts.TypeSpec, el_type_b: ts.TypeSpec):
        nonlocal is_compatible
        if isinstance(el_type_a, it_ts.IteratorType) and isinstance(el_type_b, it_ts.IteratorType):
            if not any(el_type.position_dims == "unknown" for el_type in [el_type_a, el_type_b]):
                is_compatible &= el_type_a.position_dims == el_type_b.position_dims
            is_compatible &= el_type_a.defined_dims == el_type_b.defined_dims
            is_compatible &= el_type_a.element_type == el_type_b.element_type
        else:
            is_compatible &= el_type_a == el_type_b

    type_info.apply_to_primitive_constituents(is_compatible_element, type_a, type_b)

    return is_compatible


# Problems:
#  - how to get the kind of the dimension in here? X
#    maybe directly attach the type to an axis literal?
#  - lift X (also mention to Hannes)
#  - is_compatible
#  - late offset literal in  (also mention to Hannes)
#    tests/next_tests/integration_tests/multi_feature_tests/iterator_tests/test_anton_toy.py
#  - what happens when we get a lambda function whose params are already typed
#  - write back params type in lambda
#  - documentation
#    describe why lambda can only have one type. Describe idea to solve e.g.
#     let("f", lambda x: x)(f(1)+f(1.))
#     -> let("f_int", lambda x: x, "f_float", lambda x: x)(f_int(1)+f_float(1.))
#  - make types hashable
#    - ~~either Eve with Coercion and no runtime checking,~~ dataclass hash with cached property
#  - document how scans are handled (also mention to Hannes)
#  - types are stored in the node, but will be incomplete after some passes
#  - deferred type for testing
#  - visit_FunctionDefinition


# Design decisions
#  Only the parameters of fencils need to be typed.
#  Lambda functions are not polymorphic.


def on_inferred(callback: Callable, *args: Union[ts.TypeSpec, "DeferredFunctionType"]) -> None:
    """
    Execute `callback` as soon as all `args` have a type.
    """
    ready_args = [False] * len(args)
    inferred_args = [None] * len(args)

    def mark_ready(i, type_):
        ready_args[i] = True
        inferred_args[i] = type_
        if all(ready_args):
            callback(*inferred_args)

    for i, arg in enumerate(args):
        if isinstance(arg, DeferredFunctionType):
            arg.on_type_ready(functools.partial(mark_ready, i))
        else:
            assert isinstance(arg, ts.TypeSpec)
            mark_ready(i, arg)


@dataclasses.dataclass
class DeferredFunctionType:
    """
    This class wraps a raw type inference rule to handle typing of functions.

    As functions are represented by type inference rules
    """

    #: type rule that given a set of types or type rules returns the return type or a type rule
    type_rule: rules.TypeInferenceRule
    #: offset provider used by some type rules
    offset_provider: Any
    #: node that has this type
    node: Optional[itir.Node] = None
    #: list of references to this function
    aliases: list[itir.SymRef] = dataclasses.field(default_factory=list)
    #: list of callbacks executed as soon as the type is ready
    callbacks: list[Any] = dataclasses.field(default_factory=list)
    #: the inferred type when ready and None until then
    inferred_type: Optional[ts.FunctionType] = None
    #: whether to store the type in the node or not
    store_inferred_type_in_node: bool = False

    def infer_type(
        self, return_type: ts.DataType | ts.DeferredType, *args: ts.DataType | ts.DeferredType
    ) -> ts.FunctionType:
        return ts.FunctionType(
            pos_only_args=list(args), pos_or_kw_args={}, kw_only_args={}, returns=return_type
        )

    def _infer_type_listener(self, return_type: ts.TypeSpec, *args: ts.TypeSpec) -> None:
        self.inferred_type = self.infer_type(return_type, *args)  # type: ignore[arg-type]  # ensured by assert above

        # if the type has been fully inferred, notify all `DeferredFunctionType`s that depend on it.
        for cb in self.callbacks:
            cb(self.inferred_type)

        if self.store_inferred_type_in_node:
            assert self.node
            self.node.type = self.inferred_type
            for alias in self.aliases:
                alias.type = self.inferred_type

    def on_type_ready(self, cb: Callable[[ts.TypeSpec], None]) -> None:
        if self.inferred_type:
            # type has already been inferred, just call the callback
            cb(self.inferred_type)
        else:
            self.callbacks.append(cb)

    def __call__(self, *args) -> ts.TypeSpec | rules.TypeInferenceRule:
        if "offset_provider" in inspect.signature(self.type_rule).parameters:
            return_type = self.type_rule(*args, offset_provider=self.offset_provider)
        else:
            return_type = self.type_rule(*args)

        # return type is a typing rule by itself
        if callable(return_type):
            return_type = DeferredFunctionType(
                node=None,  # node will be set by caller
                type_rule=return_type,
                offset_provider=self.offset_provider,
                store_inferred_type_in_node=True,
            )

        # delay storing the type until the return type and all arguments are inferred
        on_inferred(self._infer_type_listener, return_type, *args)

        return return_type


T = TypeVar("T", bound=itir.Node)


def _get_dimensions_from_offset_provider(offset_provider) -> dict[str, common.Dimension]:
    dimensions: dict[str, common.Dimension] = {}
    for offset_name, provider in offset_provider.items():
        dimensions[offset_name] = common.Dimension(
            value=offset_name, kind=common.DimensionKind.LOCAL
        )
        if isinstance(provider, common.Dimension):
            dimensions[provider.value] = provider
        elif isinstance(provider, common.Connectivity):
            dimensions[provider.origin_axis.value] = provider.origin_axis
            dimensions[provider.neighbor_axis.value] = provider.neighbor_axis
    return dimensions


def _get_dimensions_from_types(types) -> dict[str, common.Dimension]:
    dimensions: list[common.Dimension] = []
    for type_ in types:

        def _get_dimensions(el_type):
            if isinstance(el_type, ts.FieldType):
                dimensions.extend(el_type.dims)

        type_info.apply_to_primitive_constituents(_get_dimensions, type_)
    return {dim.value: dim for dim in dimensions}


@dataclasses.dataclass
class ITIRTypeInference(eve.NodeTranslator):
    """
    TODO
    """

    offset_provider: Any

    dimensions: dict[str, common.Dimension]

    @classmethod
    def apply(cls, node: T, *, offset_provider, inplace: bool = False) -> T:
        instance = cls(
            offset_provider=offset_provider,
            dimensions=(
                _get_dimensions_from_offset_provider(offset_provider)
                | _get_dimensions_from_types(
                    node.pre_walk_values().if_isinstance(itir.Node).getattr("type").if_is_not(None)
                )
            ),
        )
        if not inplace:
            node = copy.deepcopy(node)
        instance.visit(
            node,
            ctx={
                name: DeferredFunctionType(
                    type_rule=rules.type_inference_rules[name],
                    # builtin functions are polymorphic
                    store_inferred_type_in_node=False,
                    offset_provider=offset_provider,
                )
                for name in rules.type_inference_rules.keys()
            },
        )
        return node

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        result = super().visit(node, **kwargs)
        if isinstance(node, itir.Node):
            if isinstance(result, ts.TypeSpec):
                # TODO: verify types match
                node.type = result
            elif isinstance(result, DeferredFunctionType):
                pass
            elif callable(result):
                # TODO: only do for type rules not every callable
                return DeferredFunctionType(
                    node=node,
                    type_rule=result,
                    store_inferred_type_in_node=True,
                    offset_provider=self.offset_provider,
                )
            else:
                raise AssertionError(
                    f"Expected a 'TypeSpec' or 'DeferredFunctionType', but got "
                    f"`{type(result).__name__}`"
                )
        return result

    def visit_FencilDefinition(self, node: itir.FencilDefinition, *, ctx):
        params: dict[str, ts.DataType] = {}
        for param in node.params:
            assert isinstance(param.type, ts.DataType)
            params[param.id] = param.type

        function_definitions: dict[str, rules.TypeInferenceRule] = {}
        for fun_def in node.function_definitions:
            function_definitions[fun_def.id] = self.visit(fun_def, ctx=ctx | function_definitions)

        closures = self.visit(node.closures, ctx=ctx | params | function_definitions)
        return it_ts.FencilType(params=list(params.values()), closures=closures)

    def visit_StencilClosure(self, node: itir.StencilClosure, *, ctx):
        domain: it_ts.DomainType = self.visit(node.domain, ctx=ctx)
        inputs: list[ts.FieldType] = self.visit(node.inputs, ctx=ctx)
        output: ts.FieldType = self.visit(node.output, ctx=ctx)

        assert isinstance(domain, it_ts.DomainType)
        for output_el in type_info.primitive_constituents(output):
            assert isinstance(output_el, ts.FieldType)

        stencil_args = []
        for input_ in inputs:
            defined_dims: list[common.Dimension] | None = None

            def extract_dtype_and_defined_dims(el_type: ts.TypeSpec):
                nonlocal defined_dims
                assert isinstance(el_type, (ts.FieldType, ts.ScalarType))
                el_type = type_info.promote(el_type, always_field=True)
                if not defined_dims:
                    defined_dims = el_type.dims  # type: ignore[union-attr]  # ensured by always_field
                else:
                    # tuple inputs must all have the same defined dimensions as we
                    # create an iterator of tuples from them
                    assert defined_dims == el_type.dims  # type: ignore[union-attr]  # ensured by always_field
                return el_type.dtype  # type: ignore[union-attr]  # ensured by always_field

            element_type = type_info.apply_to_primitive_constituents(
                extract_dtype_and_defined_dims, input_
            )

            assert defined_dims is not None

            stencil_args.append(
                it_ts.IteratorType(
                    position_dims=domain.dims, defined_dims=defined_dims, element_type=element_type
                )
            )

        stencil_type_rule = self.visit(node.stencil, ctx=ctx)
        stencil_returns = stencil_type_rule(*stencil_args)

        return it_ts.StencilClosureType(
            domain=domain,
            stencil=ts.FunctionType(
                pos_only_args=stencil_args,
                pos_or_kw_args={},
                kw_only_args={},
                returns=stencil_returns,
            ),
            output=output,
            inputs=inputs,
        )

    def visit_Node(self, node: itir.Node, **kwargs):
        raise NotImplementedError(
            f"No type deduction rule for nodes of type " f"'{type(node).__name__}'."
        )

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs):
        assert (
            node.value in self.dimensions
        ), f"Dimension {node.value} not present in offset provider."
        return ts.DimensionType(dim=self.dimensions[node.value])

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs):
        # TODO: this happens in tests/next_tests/integration_tests/multi_feature_tests/iterator_tests/test_anton_toy.py
        if _is_representable_as_int(node.value):
            return it_ts.OffsetLiteralType(value=int(node.value))
        else:
            assert isinstance(node.value, str) and node.value in self.dimensions
            return it_ts.OffsetLiteralType(value=self.dimensions[node.value])

    def visit_Literal(self, node: itir.Literal, **kwargs):
        assert isinstance(node.type, ts.ScalarType)
        return node.type

    def visit_SymRef(self, node: itir.SymRef, *, ctx: dict[str, ts.TypeSpec]):
        # for testing it is useful to be able to use types without a declaration, but just storing
        # the type in the node itself.
        if node.type:
            assert node.id not in ctx or _is_compatible_type(ctx[node.id], node.type)
            return node.type
        # TODO: only allow in testing
        if node.id not in ctx:
            return ts.DeferredType(constraint=None)
        result = ctx[node.id]
        if isinstance(result, DeferredFunctionType):
            result.aliases.append(node)
        return result

    def visit_Lambda(
        self, node: itir.Lambda | itir.FunctionDefinition, *, ctx: dict[str, ts.TypeSpec]
    ):
        def fun(*args):
            return self.visit(
                node.expr, ctx=ctx | {p.id: a for p, a in zip(node.params, args, strict=True)}
            )

        return DeferredFunctionType(
            node=node,
            type_rule=fun,
            store_inferred_type_in_node=True,
            offset_provider=self.offset_provider,
        )

    visit_FunctionDefinition = visit_Lambda

    def visit_FunCall(self, node: itir.FunCall, *, ctx: dict[str, ts.TypeSpec]):
        if is_call_to(node, "cast_"):
            value, type_constructor = node.args
            assert (
                isinstance(type_constructor, itir.SymRef)
                and type_constructor.id in itir.TYPEBUILTINS
            )
            return ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

        if is_call_to(node, "tuple_get"):
            index_literal, tuple_ = node.args
            self.visit(tuple_, ctx=ctx)  # ensure tuple is typed
            assert isinstance(index_literal, itir.Literal)
            index = int(index_literal.value)
            assert isinstance(tuple_.type, ts.TupleType)
            return tuple_.type.types[index]

        fun = self.visit(node.fun, ctx=ctx)
        args = self.visit(node.args, ctx=ctx)

        result = fun(*args)

        if isinstance(result, DeferredFunctionType):
            assert not result.node
            result.node = node

        return result


infer = ITIRTypeInference.apply
