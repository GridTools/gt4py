# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import functools

from gt4py import eve
from gt4py.eve import concepts
from gt4py.eve.extended_typing import Any, Callable, Optional, TypeVar, Union
from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_call_to
from gt4py.next.iterator.type_system import type_specifications as it_ts, type_synthesizer
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.type_system.type_info import primitive_constituents


def _is_representable_as_int(s: int | str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _set_node_type(node: itir.Node, type_: ts.TypeSpec) -> None:
    if node.type:
        assert type_info.is_compatible_type(
            node.type, type_
        ), "Node already has a type which differs."
    # Also populate the type of the parameters of a lambda. That way the one can access the type
    # of a parameter by a lookup in the symbol table. As long as `_set_node_type` is used
    # exclusively, the information stays consistent with the types stored in the `FunctionType`
    # of the lambda itself.
    if isinstance(node, itir.Lambda):
        assert isinstance(type_, ts.FunctionType)
        for param, param_type in zip(node.params, type_.pos_only_args):
            _set_node_type(param, param_type)
    node.type = type_


def copy_type(from_: itir.Node, to: itir.Node, allow_untyped: bool = False) -> None:
    """
    Copy type from one node to another.

    This function mainly exists for readability reasons.
    """
    assert allow_untyped is not None or isinstance(from_.type, ts.TypeSpec)
    if from_.type is None:
        assert allow_untyped
        return
    _set_node_type(to, from_.type)


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
    ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None)

    Now, consider a simple lambda function that squares its argument using the power builtin. A
    type synthesizer for this function is simple to formulate, but merely gives us the return
    type of the function.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> square_func = im.lambda_("base")(im.call("power")("base", 2))
    >>> square_func_type_synthesizer = type_synthesizer.TypeSynthesizer(
    ...     type_synthesizer=lambda base: power(base, int_type)
    ... )
    >>> square_func_type_synthesizer(float_type, offset_provider_type={})
    ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None)

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
    >>> o_type_synthesizer(float_type, offset_provider_type={})
    ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None)
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
        offset_provider_type: common.OffsetProviderType,
    ) -> Union[ts.TypeSpec, ObservableTypeSynthesizer]:
        assert all(
            isinstance(arg, (ts.TypeSpec, ObservableTypeSynthesizer)) for arg in args
        ), "ObservableTypeSynthesizer can only be used with arguments that are TypeSpec or ObservableTypeSynthesizer"

        return_type_or_synthesizer = self.type_synthesizer(
            *args, offset_provider_type=offset_provider_type
        )

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


def _get_dimensions_from_offset_provider(
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, common.Dimension]:
    dimensions: dict[str, common.Dimension] = {}
    for offset_name, provider in offset_provider_type.items():
        dimensions[offset_name] = common.Dimension(
            value=offset_name, kind=common.DimensionKind.LOCAL
        )
        if isinstance(provider, common.Dimension):
            dimensions[provider.value] = provider
        elif isinstance(provider, common.NeighborConnectivityType):
            dimensions[provider.source_dim.value] = provider.source_dim
            dimensions[provider.codomain.value] = provider.codomain
    return dimensions


def _get_dimensions_from_types(types) -> dict[str, common.Dimension]:
    def _get_dimensions(obj: Any):
        if isinstance(obj, common.Dimension):
            yield obj
        elif isinstance(obj, ts.TypeSpec):
            for field in obj.__datamodel_fields__.keys():
                yield from _get_dimensions(getattr(obj, field))
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

    return ObservableTypeSynthesizer(
        type_synthesizer=type_synthesizer, store_inferred_type_in_node=False
    )


class SanitizeTypes(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    def visit_Node(self, node: itir.Node, *, symtable: dict[str, itir.Node]) -> itir.Node:
        node = self.generic_visit(node)
        # We only want to sanitize types that have been inferred previously such that we don't run
        # into errors because a node has been reused in a pass, but has changed type. Undeclared
        # symbols however only occur when visiting a subtree (e.g. in testing). Their types
        # can be injected by populating their type attribute, which we want to preserve here.
        is_undeclared_symbol = isinstance(node, itir.SymRef) and node.id not in symtable
        if not is_undeclared_symbol and not isinstance(node, (itir.Literal, itir.Sym)):
            node.type = None
        return node


T = TypeVar("T", bound=itir.Node)

_INITIAL_CONTEXT = {
    name: ObservableTypeSynthesizer(
        type_synthesizer=type_synthesizer.builtin_type_synthesizers[name],
        # builtin functions are polymorphic
        store_inferred_type_in_node=False,
    )
    for name in type_synthesizer.builtin_type_synthesizers.keys()
}


@dataclasses.dataclass
class ITIRTypeInference(eve.NodeTranslator):
    """
    ITIR type inference algorithm.

    See :method:ITIRTypeInference.apply for more details.
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    offset_provider_type: Optional[common.OffsetProviderType]
    #: Allow sym refs to symbols that have not been declared. Mostly used in testing.
    allow_undeclared_symbols: bool
    #: Reinference-mode skipping already typed nodes.
    reinfer: bool

    @classmethod
    def apply(
        cls,
        node: T,
        *,
        offset_provider_type: common.OffsetProviderType,
        inplace: bool = False,
        allow_undeclared_symbols: bool = False,
    ) -> T:
        """
        Infer the type of ``node`` and its sub-nodes.

        Arguments:
            node: The :class:`itir.Node` to infer the types of.

        Keyword Arguments:
            offset_provider_type: Offset provider dictionary.
            inplace: Write types directly to the given ``node`` instead of returning a copy.
            allow_undeclared_symbols: Allow references to symbols that don't have a corresponding
              declaration. This is useful for testing or inference on partially inferred sub-nodes.

        Preconditions:

        All parameters in :class:`itir.Program` must have a type
        defined, as they are the starting point for type propagation.

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
        node = SanitizeTypes().visit(node)

        if isinstance(node, itir.Program):
            assert all(isinstance(param.type, ts.DataType) for param in node.params), (
                "All parameters in 'itir.Program' must have a type "
                "defined, as they are the starting point for type propagation.",
            )

        instance = cls(
            offset_provider_type=offset_provider_type,
            allow_undeclared_symbols=allow_undeclared_symbols,
            reinfer=False,
        )
        if not inplace:
            node = copy.deepcopy(node)
        instance.visit(node, ctx=_INITIAL_CONTEXT)
        return node

    @classmethod
    def apply_reinfer(cls, node: T) -> T:
        """
        Given a partially typed node infer the type of ``node`` and its sub-nodes.

        Contrary to the regular inference, this method does not descend into already typed sub-nodes
        and can be used as a lightweight way to restore type information during a pass.

        Note that this function alters the input node, which is usually desired, and more
        performant.

        Arguments:
            node: The :class:`itir.Node` to infer the types of.
        """
        if node.type:  # already inferred
            return node

        instance = cls(offset_provider_type=None, allow_undeclared_symbols=True, reinfer=True)
        instance.visit(node, ctx=_INITIAL_CONTEXT)
        return node

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        # we found a node that is typed, do not descend into children
        if self.reinfer and isinstance(node, itir.Node) and node.type:
            if isinstance(node.type, ts.FunctionType):
                return _type_synthesizer_from_function_type(node.type)
            return node.type

        result = super().visit(node, **kwargs)

        if isinstance(node, itir.Node):
            if isinstance(result, ts.TypeSpec):
                if node.type and not isinstance(node.type, ts.DeferredType):
                    assert type_info.is_compatible_type(node.type, result)
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
        assert domain.dims != "unknown"
        assert node.dtype
        return type_info.apply_to_primitive_constituents(
            lambda dtype: ts.FieldType(dims=domain.dims, dtype=dtype),
            node.dtype,
        )

    def visit_IfStmt(self, node: itir.IfStmt, *, ctx) -> None:
        cond = self.visit(node.cond, ctx=ctx)
        assert cond == ts.ScalarType(kind=ts.ScalarKind.BOOL)
        self.visit(node.true_branch, ctx=ctx)
        self.visit(node.false_branch, ctx=ctx)

    def visit_SetAt(self, node: itir.SetAt, *, ctx) -> None:
        self.visit(node.expr, ctx=ctx)
        self.visit(node.domain, ctx=ctx)
        self.visit(node.target, ctx=ctx)
        assert node.target.type is not None and node.expr.type is not None
        for target_type, path in primitive_constituents(node.target.type, with_path_arg=True):
            # the target can have fewer elements than the expr in which case the output from the
            # expression is simply discarded.
            expr_type = functools.reduce(
                lambda tuple_type, i: tuple_type.types[i]  # type: ignore[attr-defined]  # format ensured by primitive_constituents
                # `ts.DeferredType` only occurs for scans returning a tuple
                if not isinstance(tuple_type, ts.DeferredType)
                else ts.DeferredType(constraint=None),
                path,
                node.expr.type,
            )
            assert isinstance(target_type, (ts.FieldType, ts.DeferredType))
            assert isinstance(expr_type, (ts.FieldType, ts.DeferredType))
            # TODO(tehrengruber): The lowering emits domains that always have the horizontal domain
            #  first. Since the expr inherits the ordering from the domain this can lead to a mismatch
            #  between the target and expr (e.g. when the target has dimension K, Vertex). We should
            #  probably just change the behaviour of the lowering. Until then we do this more
            #  complicated comparison.
            if isinstance(target_type, ts.FieldType) and isinstance(expr_type, ts.FieldType):
                assert (
                    set(expr_type.dims).issubset(set(target_type.dims))
                    and target_type.dtype == expr_type.dtype
                )

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs) -> ts.DimensionType:
        return ts.DimensionType(dim=common.Dimension(value=node.value, kind=node.kind))

    # TODO: revisit what we want to do with OffsetLiterals as we already have an Offset type in
    #  the frontend.
    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs) -> it_ts.OffsetLiteralType:
        if _is_representable_as_int(node.value):
            return it_ts.OffsetLiteralType(
                value=ts.ScalarType(
                    kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper())
                )
            )
        else:
            assert isinstance(node.value, str)
            return it_ts.OffsetLiteralType(value=node.value)

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
            self.visit(value, ctx=ctx)  # ensure types in value are also inferred
            assert (
                isinstance(type_constructor, itir.SymRef)
                and type_constructor.id in builtins.TYPE_BUILTINS
            )
            return ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

        if is_call_to(node, "tuple_get"):
            index_literal, tuple_ = node.args
            self.visit(tuple_, ctx=ctx)  # ensure tuple is typed
            assert isinstance(index_literal, itir.Literal)
            index = int(index_literal.value)
            if isinstance(tuple_.type, ts.DeferredType):
                return ts.DeferredType(constraint=None)
            assert isinstance(tuple_.type, ts.TupleType)
            return tuple_.type.types[index]

        fun = self.visit(node.fun, ctx=ctx)
        args = self.visit(node.args, ctx=ctx)

        result = fun(*args, offset_provider_type=self.offset_provider_type)

        if isinstance(result, ObservableTypeSynthesizer):
            assert not result.node
            result.node = node

        return result

    def visit_Node(self, node: itir.Node, **kwargs):
        raise NotImplementedError(f"No type rule for nodes of type " f"'{type(node).__name__}'.")


infer = ITIRTypeInference.apply

reinfer = ITIRTypeInference.apply_reinfer
