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

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import functools

from gt4py import eve
from gt4py.eve import concepts
from gt4py.eve.extended_typing import Any, Callable, Optional, TypeVar, Union
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_call_to
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.iterator.type_system import type_specifications as it_ts, type_synthesizer
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.type_system.type_info import primitive_constituents


def _is_representable_as_int(s: int | str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_compatible_type(type_a: ts.TypeSpec, type_b: ts.TypeSpec):
    """
    Predicate to determine if two types are compatible.

    This function gracefully handles:
     - iterators with unknown positions which are considered compatible to any other positions
       of another iterator.
     - iterators which are defined everywhere, i.e. empty defined dimensions
    Beside that this function simply checks for equality of types.

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

    if isinstance(type_a, it_ts.IteratorType) and isinstance(type_b, it_ts.IteratorType):
        if not any(el_type.position_dims == "unknown" for el_type in [type_a, type_b]):
            is_compatible &= type_a.position_dims == type_b.position_dims
        if type_a.defined_dims and type_b.defined_dims:
            is_compatible &= type_a.defined_dims == type_b.defined_dims
        is_compatible &= type_a.element_type == type_b.element_type
    elif isinstance(type_a, ts.TupleType) and isinstance(type_b, ts.TupleType):
        for el_type_a, el_type_b in zip(type_a.types, type_b.types, strict=True):
            is_compatible &= _is_compatible_type(el_type_a, el_type_b)
    elif isinstance(type_a, ts.FunctionType) and isinstance(type_b, ts.FunctionType):
        for arg_a, arg_b in zip(type_a.pos_only_args, type_b.pos_only_args, strict=True):
            is_compatible &= _is_compatible_type(arg_a, arg_b)
        for arg_a, arg_b in zip(
            type_a.pos_or_kw_args.values(), type_b.pos_or_kw_args.values(), strict=True
        ):
            is_compatible &= _is_compatible_type(arg_a, arg_b)
        for arg_a, arg_b in zip(
            type_a.kw_only_args.values(), type_b.kw_only_args.values(), strict=True
        ):
            is_compatible &= _is_compatible_type(arg_a, arg_b)
        is_compatible &= _is_compatible_type(type_a.returns, type_b.returns)
    else:
        is_compatible &= type_a == type_b

    return is_compatible


def _set_node_type(node: itir.Node, type_: ts.TypeSpec) -> None:
    if node.type:
        assert _is_compatible_type(node.type, type_), "Node already has a type which differs."
    node.type = type_


def on_inferred(callback: Callable, *args: Union[ts.TypeSpec, ObservableTypeSynthesizer]) -> None:
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
        if isinstance(arg, ObservableTypeSynthesizer):
            arg.on_type_ready(functools.partial(mark_ready, i))
        else:
            assert isinstance(arg, ts.TypeSpec)
            mark_ready(i, arg)


@dataclasses.dataclass
class ObservableTypeSynthesizer(type_synthesizer.TypeSynthesizer):
    """
    This class wraps a type synthesizer to handle typing of nodes representing functions.

    The type inference algorithm represents functions as type synthesizer, i.e. regular
    callables that given a set of arguments compute / deduce the return type. The return type of
    functions, let it be a builtin like ``itir.plus`` or a user defined lambda function, is only
    defined when all its arguments are typed.

    Let's start with a small example to exemplify this. The power function has a rather simple
    type synthesizer, where the output type is simply the type of the base.

    >>> def power(base: ts.ScalarType, exponent: ts.ScalarType) -> ts.ScalarType:
    ...     return base
    >>> float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    >>> int_type = ts.ScalarType(kind=ts.ScalarKind.INT64)
    >>> power(float_type, int_type)
    ScalarType(kind=<ScalarKind.FLOAT64: 1064>, shape=None)

    Now, consider a simple lambda function that squares its argument using the power builtin. A
    type synthesizer for this function is simple to formulate, but merely gives us the return
    type of the function.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> square_func = im.lambda_("base")(im.call("power")("base", 2))
    >>> square_func_type_synthesizer = type_synthesizer.TypeSynthesizer(
    ...     type_synthesizer=lambda base: power(base, int_type)
    ... )
    >>> square_func_type_synthesizer(float_type, offset_provider={})
    ScalarType(kind=<ScalarKind.FLOAT64: 1064>, shape=None)

    Note that without a corresponding call the function itself can not be fully typed and as such
    the type inference algorithm has to defer typing until then. This task is handled transparently
    (in the sense that an ``ObservableTypeSynthesizer`` is a type synthesizer again) by this
    class. Given a type synthesizer and a node we obtain a new type synthesizer that when
    evaluated stores the type of the function in the node.

    >>> o_type_synthesizer = ObservableTypeSynthesizer(
    ...     type_synthesizer=square_func_type_synthesizer,
    ...     node=square_func,
    ...     store_inferred_type_in_node=True,
    ... )
    >>> o_type_synthesizer(float_type, offset_provider={})
    ScalarType(kind=<ScalarKind.FLOAT64: 1064>, shape=None)
    >>> square_func.type == ts.FunctionType(
    ...     pos_only_args=[float_type], pos_or_kw_args={}, kw_only_args={}, returns=float_type
    ... )
    True

    Note that this is a simple example where the type of the arguments and the return value is
    available when the function is called. In order to support higher-order functions, where
    arguments or return value are functions itself (i.e. passed as type rules) this class provides
    additional functionality for multiple typing rules to notify each other about a type being
    ready.
    """

    #: node that has this type
    node: Optional[itir.Node] = None
    #: list of references to this function
    aliases: list[itir.SymRef] = dataclasses.field(default_factory=list)
    #: list of callbacks executed as soon as the type is ready
    callbacks: list[Callable[[ts.TypeSpec], None]] = dataclasses.field(default_factory=list)
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

        # if the type has been fully inferred, notify all `ObservableTypeSynthesizer`s that depend on it.
        for cb in self.callbacks:
            cb(self.inferred_type)

        if self.store_inferred_type_in_node:
            assert self.node
            _set_node_type(self.node, self.inferred_type)
            self.node.type = self.inferred_type
            for alias in self.aliases:
                _set_node_type(alias, self.inferred_type)

    def on_type_ready(self, cb: Callable[[ts.TypeSpec], None]) -> None:
        if self.inferred_type:
            # type has already been inferred, just call the callback
            cb(self.inferred_type)
        else:
            self.callbacks.append(cb)

    def __call__(
        self,
        *args: type_synthesizer.TypeOrTypeSynthesizer,
        offset_provider: common.OffsetProvider,
    ) -> Union[ts.TypeSpec, ObservableTypeSynthesizer]:
        assert all(
            isinstance(arg, (ts.TypeSpec, ObservableTypeSynthesizer)) for arg in args
        ), "ObservableTypeSynthesizer can only be used with arguments that are TypeSpec or ObservableTypeSynthesizer"

        return_type_or_synthesizer = self.type_synthesizer(*args, offset_provider=offset_provider)

        # return type is a typing rule by itself
        if isinstance(return_type_or_synthesizer, type_synthesizer.TypeSynthesizer):
            return_type_or_synthesizer = ObservableTypeSynthesizer(
                node=None,  # node will be set by caller
                type_synthesizer=return_type_or_synthesizer,
                store_inferred_type_in_node=True,
            )

        assert isinstance(return_type_or_synthesizer, (ts.TypeSpec, ObservableTypeSynthesizer))

        # delay storing the type until the return type and all arguments are inferred
        on_inferred(self._infer_type_listener, return_type_or_synthesizer, *args)  # type: ignore[arg-type] # ensured by assert above

        return return_type_or_synthesizer


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
    def _get_dimensions(obj: Any):
        if isinstance(obj, common.Dimension):
            yield obj
        elif isinstance(obj, ts.TypeSpec):
            for field in dataclasses.fields(obj.__class__):
                yield from _get_dimensions(getattr(obj, field.name))
        elif isinstance(obj, collections.abc.Mapping):
            for el in obj.values():
                yield from _get_dimensions(el)
        elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
            for el in obj:
                yield from _get_dimensions(el)

    return {dim.value: dim for dim in _get_dimensions(types)}


def _type_synthesizer_from_function_type(fun_type: ts.FunctionType):
    def type_synthesizer(*args, **kwargs):
        assert type_info.accepts_args(fun_type, with_args=list(args), with_kwargs=kwargs)
        return fun_type.returns

    return type_synthesizer


class RemoveTypes(eve.NodeTranslator):
    def visit_Node(self, node: itir.Node):
        node = self.generic_visit(node)
        if not isinstance(node, (itir.Literal, itir.Sym)):
            node.type = None
        return node


T = TypeVar("T", bound=itir.Node)


@dataclasses.dataclass
class ITIRTypeInference(eve.NodeTranslator):
    """
    ITIR type inference algorithm.

    See :method:ITIRTypeInference.apply for more details.
    """

    offset_provider: common.OffsetProvider
    #: Mapping from a dimension name to the actual dimension instance.
    dimensions: dict[str, common.Dimension]
    #: Allow sym refs to symbols that have not been declared. Mostly used in testing.
    allow_undeclared_symbols: bool

    @classmethod
    def apply(
        cls,
        node: T,
        *,
        offset_provider: common.OffsetProvider,
        inplace: bool = False,
        allow_undeclared_symbols: bool = False,
    ) -> T:
        """
        Infer the type of ``node`` and its sub-nodes.

        Arguments:
            node: The :class:`itir.Node` to infer the types of.

        Keyword Arguments:
            offset_provider: Offset provider dictionary.
            inplace: Write types directly to the given ``node`` instead of returning a copy.
            allow_undeclared_symbols: Allow references to symbols that don't have a corresponding
              declaration. This is useful for testing or inference on partially inferred sub-nodes.

        Design decisions:
        - Lamba functions are monomorphic
        Builtin functions like ``plus`` are by design polymorphic and only their argument and return
        types are of importance in transformations. Lambda functions on the contrary also have a
        body on which we would like to run transformations. By choosing them to be monomorphic all
        types in the body can be inferred to a concrete type, making reasoning about them in
        transformations simple. Consequently, the following is invalid as `f` is called with
        arguments of different type
        ```
        let f = λ(a) → a+a
            in f(1)+f(1.)
        ```
        In case we want polymorphic lambda functions, i.e. generic functions in the frontend
        could be implemented that way, current consensus is to instead implement a transformation
        that duplicates the lambda function for each of the types it is called with
        ```
        let f_int = λ(a) → a+a, f_float = λ(a) → a+a
            in f_int(1)+f_float(1.)
        ```
        Note that this is not the only possible choice. Polymorphic lambda functions and a type
        inference algorithm that only infers the most generic type would allow us to run
        transformations without this duplication and reduce code size early. However, this would
        require careful planning and documentation on what information a transformation needs.

        Limitations:

        - The current position of (iterator) arguments to a lifted stencil is unknown
        Consider the following trivial stencil: ``λ(it) → deref(it)``. A priori we don't know
        what the current position of ``it`` is (inside the body of the lambda function), but only
        when we call the stencil with an actual iterator the position becomes known. Consequently,
        when we lift the stencil, the position of its iterator arguments is only known as soon as
        the iterator as returned by the applied lift is dereferenced. Deferring the inference
        of the current position for lifts has been decided to be too complicated as we don't need
        the information right now and is hence not implemented.

        - Iterators only ever reference values, not columns.
        The initial version of the ITIR used iterators of columns and vectorized operations between
        columns in order to express scans. This differentiation is not needed in our transformations
        and as such was not implemented here.
        """
        # TODO(tehrengruber): some of the transformations reuse nodes with type information that
        #  becomes invalid (e.g. the shift part of ``shift(...)(it)`` has a different type when used
        #  on a different iterator). For now we just delete all types in case we are working an
        #   parts of a program.
        if not allow_undeclared_symbols:
            node = RemoveTypes().visit(node)

        instance = cls(
            offset_provider=offset_provider,
            dimensions=(
                _get_dimensions_from_offset_provider(offset_provider)
                | _get_dimensions_from_types(
                    node.pre_walk_values()
                    .if_isinstance(itir.Node)
                    .getattr("type")
                    .if_is_not(None)
                    .to_list()
                )
            ),
            allow_undeclared_symbols=allow_undeclared_symbols,
        )
        if not inplace:
            node = copy.deepcopy(node)
        instance.visit(
            node,
            ctx={
                name: ObservableTypeSynthesizer(
                    type_synthesizer=type_synthesizer.builtin_type_synthesizers[name],
                    # builtin functions are polymorphic
                    store_inferred_type_in_node=False,
                )
                for name in type_synthesizer.builtin_type_synthesizers.keys()
            },
        )
        return node

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        result = super().visit(node, **kwargs)
        if isinstance(node, itir.Node):
            if isinstance(result, ts.TypeSpec):
                if node.type:
                    assert _is_compatible_type(node.type, result)
                node.type = result
            elif isinstance(result, ObservableTypeSynthesizer) or result is None:
                pass
            elif isinstance(result, type_synthesizer.TypeSynthesizer):
                # this case occurs either when a Lambda node is visited or TypeSynthesizer returns
                # another type synthesizer.
                return ObservableTypeSynthesizer(
                    node=node,
                    type_synthesizer=result,
                    store_inferred_type_in_node=True,
                )
            else:
                raise AssertionError(
                    f"Expected a 'TypeSpec', `TypeSynthesizer` or 'ObservableTypeSynthesizer', "
                    f"`but got {type(result).__name__}`"
                )
        return result

    # TODO(tehrengruber): Remove after new ITIR format with apply_stencil is used everywhere
    def visit_FencilDefinition(self, node: itir.FencilDefinition, *, ctx) -> it_ts.FencilType:
        params: dict[str, ts.DataType] = {}
        for param in node.params:
            assert isinstance(param.type, ts.DataType)
            params[param.id] = param.type

        function_definitions: dict[str, type_synthesizer.TypeSynthesizer] = {}
        for fun_def in node.function_definitions:
            function_definitions[fun_def.id] = self.visit(fun_def, ctx=ctx | function_definitions)

        closures = self.visit(node.closures, ctx=ctx | params | function_definitions)
        return it_ts.FencilType(params=params, closures=closures)

    # TODO(tehrengruber): Remove after new ITIR format with apply_stencil is used everywhere
    def visit_FencilWithTemporaries(
        self, node: global_tmps.FencilWithTemporaries, *, ctx
    ) -> it_ts.FencilType:
        # TODO(tehrengruber): This implementation is not very appealing. Since we are about to
        #  refactor the IR anyway this is fine for now.
        params: dict[str, ts.DataType] = {}
        for param in node.params:
            assert isinstance(param.type, ts.DataType)
            params[param.id] = param.type
        # infer types of temporary declarations
        tmps: dict[str, ts.FieldType] = {}
        for tmp_node in node.tmps:
            tmps[tmp_node.id] = self.visit(tmp_node, ctx=ctx | params)
        # and store them in the inner fencil
        for fencil_param in node.fencil.params:
            if fencil_param.id in tmps:
                fencil_param.type = tmps[fencil_param.id]
        self.visit(node.fencil, ctx=ctx)
        assert isinstance(node.fencil.type, it_ts.FencilType)
        return node.fencil.type

    def visit_Program(self, node: itir.Program, *, ctx) -> it_ts.ProgramType:
        params: dict[str, ts.DataType] = {}
        for param in node.params:
            assert isinstance(param.type, ts.DataType)
            params[param.id] = param.type
        decls: dict[str, ts.FieldType] = {}
        for fun_def in node.function_definitions:
            decls[fun_def.id] = self.visit(fun_def, ctx=ctx | params | decls)
        for decl_node in node.declarations:
            decls[decl_node.id] = self.visit(decl_node, ctx=ctx | params | decls)
        self.visit(node.body, ctx=ctx | params | decls)
        return it_ts.ProgramType(params=params)

    def visit_Temporary(self, node: itir.Temporary, *, ctx) -> ts.FieldType | ts.TupleType:
        domain = self.visit(node.domain, ctx=ctx)
        assert isinstance(domain, it_ts.DomainType)
        assert node.dtype
        return type_info.apply_to_primitive_constituents(
            lambda dtype: ts.FieldType(dims=domain.dims, dtype=dtype), node.dtype
        )

    def visit_SetAt(self, node: itir.SetAt, *, ctx) -> None:
        self.visit(node.expr, ctx=ctx)
        self.visit(node.domain, ctx=ctx)
        self.visit(node.target, ctx=ctx)
        assert node.target.type is not None and node.expr.type is not None
        for target_type, path in primitive_constituents(node.target.type, with_path_arg=True):
            # the target can have fewer elements than the expr in which case the output from the
            # expression is simply discarded.
            expr_type = functools.reduce(
                lambda tuple_type, i: tuple_type.types[i],  # type: ignore[attr-defined]  # format ensured by primitive_constituents
                path,
                node.expr.type,
            )
            assert isinstance(target_type, ts.FieldType)
            assert isinstance(expr_type, ts.FieldType)
            # TODO(tehrengruber): The lowering emits domains that always have the horizontal domain
            #  first. Since the expr inherits the ordering from the domain this can lead to a mismatch
            #  between the target and expr (e.g. when the target has dimension K, Vertex). We should
            #  probably just change the behaviour of the lowering. Until then we do this more
            #  complicated comparison.
            assert (
                set(expr_type.dims) == set(target_type.dims)
                and target_type.dtype == expr_type.dtype
            )

    # TODO(tehrengruber): Remove after new ITIR format with apply_stencil is used everywhere
    def visit_StencilClosure(self, node: itir.StencilClosure, *, ctx) -> it_ts.StencilClosureType:
        domain: it_ts.DomainType = self.visit(node.domain, ctx=ctx)
        inputs: list[ts.FieldType] = self.visit(node.inputs, ctx=ctx)
        output: ts.FieldType = self.visit(node.output, ctx=ctx)

        assert isinstance(domain, it_ts.DomainType)
        for output_el in type_info.primitive_constituents(output):
            assert isinstance(output_el, ts.FieldType)

        stencil_type_synthesizer = self.visit(node.stencil, ctx=ctx)
        stencil_args = [
            type_synthesizer._convert_as_fieldop_input_to_iterator(domain, input_)
            for input_ in inputs
        ]
        stencil_returns = stencil_type_synthesizer(
            *stencil_args, offset_provider=self.offset_provider
        )

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

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs) -> ts.DimensionType:
        assert (
            node.value in self.dimensions
        ), f"Dimension {node.value} not present in offset provider."
        return ts.DimensionType(dim=self.dimensions[node.value])

    # TODO: revisit what we want to do with OffsetLiterals as we already have an Offset type in
    #  the frontend.
    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs) -> it_ts.OffsetLiteralType:
        if _is_representable_as_int(node.value):
            return it_ts.OffsetLiteralType(
                value=ts.ScalarType(kind=getattr(ts.ScalarKind, itir.INTEGER_INDEX_BUILTIN.upper()))
            )
        else:
            assert isinstance(node.value, str) and node.value in self.dimensions
            return it_ts.OffsetLiteralType(value=self.dimensions[node.value])

    def visit_Literal(self, node: itir.Literal, **kwargs) -> ts.ScalarType:
        assert isinstance(node.type, ts.ScalarType)
        return node.type

    def visit_SymRef(
        self, node: itir.SymRef, *, ctx: dict[str, ts.TypeSpec]
    ) -> ts.TypeSpec | type_synthesizer.TypeSynthesizer:
        # for testing, it is useful to be able to use types without a declaration
        if self.allow_undeclared_symbols and node.id not in ctx:
            # type has been stored in the node itself
            if node.type:
                if isinstance(node.type, ts.FunctionType):
                    return _type_synthesizer_from_function_type(node.type)
                return node.type
            return ts.DeferredType(constraint=None)
        assert node.id in ctx
        result = ctx[node.id]
        if isinstance(result, ObservableTypeSynthesizer):
            result.aliases.append(node)
        return result

    def visit_Lambda(
        self, node: itir.Lambda | itir.FunctionDefinition, *, ctx: dict[str, ts.TypeSpec]
    ) -> type_synthesizer.TypeSynthesizer:
        @type_synthesizer.TypeSynthesizer
        def fun(*args):
            return self.visit(
                node.expr, ctx=ctx | {p.id: a for p, a in zip(node.params, args, strict=True)}
            )

        return fun

    visit_FunctionDefinition = visit_Lambda

    def visit_FunCall(
        self, node: itir.FunCall, *, ctx: dict[str, ts.TypeSpec]
    ) -> ts.TypeSpec | type_synthesizer.TypeSynthesizer:
        # grammar builtins
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

        result = fun(*args, offset_provider=self.offset_provider)

        if isinstance(result, ObservableTypeSynthesizer):
            assert not result.node
            result.node = node

        return result

    def visit_Node(self, node: itir.Node, **kwargs):
        raise NotImplementedError(f"No type rule for nodes of type " f"'{type(node).__name__}'.")


infer = ITIRTypeInference.apply
