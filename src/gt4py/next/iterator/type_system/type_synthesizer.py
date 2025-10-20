# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import TypeVar, cast, overload

from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import Callable, Iterable, Optional, Union
from gt4py.next import common, utils
from gt4py.next.ffront import fbuiltins
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.utils import tree_map


def _type_synth_arg_cache_key(type_or_synth: TypeOrTypeSynthesizer) -> int:
    if isinstance(type_or_synth, TypeSynthesizer):
        return id(type_or_synth)
    # TODO(tehrengruber): use regular __hash__ again when ts.TypeSpec supports it.
    return hash(eve_utils.content_hash(type_or_synth))


@dataclasses.dataclass
class TypeSynthesizer:
    """
    Callable that given the type of the arguments to a function derives its return type.

    In case the function is a higher-order function the returned value is not a type, but another
    function type-synthesizer.

    In addition to the derivation of the return type a function type-synthesizer can perform checks
    on the argument types.

    The motivation for this class instead of a simple callable is to allow
     - isinstance checks to determine if an object is actually (meant to be) a type
       synthesizer and not just any callable.
     - writing simple type synthesizers without cluttering the signature with the additional
       offset_provider_type argument that is only needed by some.
    """

    type_synthesizer: Callable[..., TypeOrTypeSynthesizer]
    cache: bool = False

    def __post_init__(self):
        if "offset_provider_type" not in inspect.signature(self.type_synthesizer).parameters:
            synthesizer = self.type_synthesizer
            self.type_synthesizer = lambda *args, offset_provider_type, **kwargs: synthesizer(
                *args, **kwargs
            )
        if self.cache:
            self.type_synthesizer = eve_utils.lru_cache(
                self.type_synthesizer,
                # we only cache `itir.Lambda` right now which is only ever evaluated with
                # the same arguments. Hence, the cache only needs a size of one.
                maxsize=1,
                key=_type_synth_arg_cache_key,
            )

    def __call__(
        self,
        *args: TypeOrTypeSynthesizer,
        offset_provider_type: common.OffsetProviderType,
        **kwargs,
    ) -> TypeOrTypeSynthesizer:
        return self.type_synthesizer(*args, offset_provider_type=offset_provider_type, **kwargs)


TypeOrTypeSynthesizer = Union[ts.TypeSpec, TypeSynthesizer]
F = TypeVar("F", bound=Callable[..., TypeOrTypeSynthesizer])


@overload
def type_synthesizer(fun: F) -> TypeSynthesizer: ...
@overload
def type_synthesizer(*, cache: bool = False) -> Callable[[F], TypeSynthesizer]: ...


def type_synthesizer(
    fun: Optional[F] = None, cache: bool = False
) -> Union[TypeSynthesizer, Callable[[F], TypeSynthesizer]]:
    if fun is None:
        return functools.partial(TypeSynthesizer, cache=cache)
    return TypeSynthesizer(fun, cache=cache)


#: dictionary from name of a builtin to its type synthesizer
builtin_type_synthesizers: dict[str, TypeSynthesizer] = {}


def _is_derefable_iterator_type(it_type: it_ts.IteratorType, *, default: bool = True) -> bool:
    # for an iterator with unknown position we can not tell if it is derefable,
    # so we just return the default
    if it_type.position_dims == "unknown":
        return default
    return set(it_type.defined_dims).issubset(set(it_type.position_dims))


def _register_builtin_type_synthesizer(
    synthesizer: Optional[Callable[..., TypeOrTypeSynthesizer]] = None,
    *,
    fun_names: Optional[Iterable[str]] = None,
):
    if synthesizer is None:
        return functools.partial(_register_builtin_type_synthesizer, fun_names=fun_names)

    # store names in function object for better debuggability
    synthesizer.fun_names = fun_names or [synthesizer.__name__]  # type: ignore[attr-defined]
    for f in synthesizer.fun_names:  # type: ignore[attr-defined]
        builtin_type_synthesizers[f] = type_synthesizer(synthesizer)
    return synthesizer


@_register_builtin_type_synthesizer(
    fun_names=builtins.UNARY_MATH_NUMBER_BUILTINS | builtins.UNARY_MATH_FP_BUILTINS
)
def _(val: ts.ScalarType) -> ts.ScalarType:
    return val


@_register_builtin_type_synthesizer
def power(base: ts.ScalarType, exponent: ts.ScalarType) -> ts.ScalarType:
    return base


@_register_builtin_type_synthesizer(fun_names=builtins.BINARY_MATH_NUMBER_BUILTINS)
def _(lhs: ts.ScalarType, rhs: ts.ScalarType) -> ts.ScalarType:
    if isinstance(lhs, ts.DeferredType):
        return rhs
    if isinstance(rhs, ts.DeferredType):
        return lhs
    assert lhs == rhs
    return lhs


@_register_builtin_type_synthesizer(
    fun_names=builtins.UNARY_MATH_FP_PREDICATE_BUILTINS | builtins.UNARY_LOGICAL_BUILTINS
)
def _(arg: ts.ScalarType) -> ts.ScalarType:
    return ts.ScalarType(kind=ts.ScalarKind.BOOL)


def synthesize_binary_math_comparison_builtins(
    lhs, rhs
) -> ts.ScalarType | ts.TupleType | ts.DomainType:
    if isinstance(lhs, ts.ScalarType) and isinstance(rhs, ts.DimensionType):
        return ts.DomainType(dims=[rhs.dim])
    if isinstance(lhs, ts.DimensionType) and isinstance(rhs, ts.ScalarType):
        return ts.DomainType(dims=[lhs.dim])
    assert all(isinstance(lhs, (ts.ScalarType, ts.DeferredType)) for arg in (lhs, rhs))
    return ts.ScalarType(kind=ts.ScalarKind.BOOL)


@_register_builtin_type_synthesizer(fun_names=builtins.BINARY_MATH_COMPARISON_BUILTINS)
def _(lhs, rhs) -> ts.ScalarType | ts.TupleType | ts.DomainType:
    return synthesize_binary_math_comparison_builtins(lhs, rhs)


@_register_builtin_type_synthesizer(fun_names=builtins.BINARY_LOGICAL_BUILTINS)
def _(lhs, rhs) -> ts.ScalarType | ts.TupleType | ts.DomainType:
    if isinstance(lhs, ts.DomainType) and isinstance(rhs, ts.DomainType):
        assert lhs.dims != "unknown" and rhs.dims != "unknown"
        return ts.DomainType(dims=common.promote_dims(lhs.dims, rhs.dims))
    else:
        return synthesize_binary_math_comparison_builtins(lhs, rhs)


@_register_builtin_type_synthesizer
def deref(it: it_ts.IteratorType | ts.DeferredType) -> ts.DataType | ts.DeferredType:
    if isinstance(it, ts.DeferredType):
        return ts.DeferredType(constraint=None)
    assert isinstance(it, it_ts.IteratorType)
    assert _is_derefable_iterator_type(it)
    return it.element_type


@_register_builtin_type_synthesizer
def can_deref(it: it_ts.IteratorType | ts.DeferredType) -> ts.ScalarType:
    assert isinstance(it, ts.DeferredType) or isinstance(it, it_ts.IteratorType)
    # note: We don't check if the iterator is derefable here as the iterator only needs to
    # to have a valid position. Consider a nested reduction, e.g.
    #  `reduce(plus, 0)(neighbors(V2Eₒ, (↑(λ(a) → reduce(plus, 0)(neighbors(E2Vₒ, a))))(it))`
    # When written using a `can_deref` we only care if the edge neighbor of the vertex of `it`
    # is valid, i.e. we want `can_deref(shift(V2Eₒ, i)(it))` to return true. But since `it` is an
    # iterator backed by a vertex field, the iterator is not derefable in the sense that
    # its position is a valid position of the backing field.
    # TODO(tehrengruber): Consider renaming can_deref to something that better reflects its
    #  meaning.
    return ts.ScalarType(kind=ts.ScalarKind.BOOL)


@_register_builtin_type_synthesizer
def if_(
    pred: ts.ScalarType | ts.DeferredType, true_branch: ts.DataType, false_branch: ts.DataType
) -> ts.DataType:
    if isinstance(true_branch, ts.TupleType) and isinstance(false_branch, ts.TupleType):
        return tree_map(
            collection_type=ts.TupleType,
            result_collection_constructor=lambda _, elts: ts.TupleType(types=[*elts]),
        )(functools.partial(if_, pred))(true_branch, false_branch)

    assert not isinstance(true_branch, ts.TupleType) and not isinstance(false_branch, ts.TupleType)
    assert isinstance(pred, ts.DeferredType) or (
        isinstance(pred, ts.ScalarType) and pred.kind == ts.ScalarKind.BOOL
    )
    # TODO(tehrengruber): Enable this or a similar check. In case the true- and false-branch are
    #  iterators defined on different positions this fails. For the GTFN backend we also don't
    #  want this, but for roundtrip it is totally fine.
    # assert true_branch == false_branch  # noqa: ERA001

    return true_branch


@_register_builtin_type_synthesizer
def make_const_list(scalar: ts.ScalarType) -> ts.ListType:
    assert isinstance(scalar, ts.ScalarType)
    return ts.ListType(element_type=scalar)


@_register_builtin_type_synthesizer
def list_get(index: ts.ScalarType | it_ts.OffsetLiteralType, list_: ts.ListType) -> ts.DataType:
    if isinstance(index, it_ts.OffsetLiteralType):
        assert isinstance(index.value, ts.ScalarType)
        index = index.value
    assert isinstance(index, ts.ScalarType) and type_info.is_integral(index)
    assert isinstance(list_, ts.ListType)
    return list_.element_type


@_register_builtin_type_synthesizer
def named_range(
    dim: ts.DimensionType, start: ts.ScalarType, stop: ts.ScalarType
) -> it_ts.NamedRangeType:
    assert isinstance(dim, ts.DimensionType)
    return it_ts.NamedRangeType(dim=dim.dim)


@_register_builtin_type_synthesizer(fun_names=["cartesian_domain", "unstructured_domain"])
def _(*args: it_ts.NamedRangeType) -> ts.DomainType:
    assert all(isinstance(arg, it_ts.NamedRangeType) for arg in args)
    return ts.DomainType(dims=[arg.dim for arg in args])


@_register_builtin_type_synthesizer
def make_tuple(*args: ts.DataType) -> ts.TupleType:
    return ts.TupleType(types=list(args))


@_register_builtin_type_synthesizer
def index(arg: ts.DimensionType) -> ts.FieldType:
    return ts.FieldType(
        dims=[arg.dim],
        dtype=ts.ScalarType(kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper())),
    )


@_register_builtin_type_synthesizer
def concat_where(
    domain: ts.DomainType,
    true_field: ts.FieldType | ts.TupleType | ts.DeferredType,
    false_field: ts.FieldType | ts.TupleType | ts.DeferredType,
) -> ts.FieldType | ts.TupleType | ts.DeferredType:
    if isinstance(true_field, ts.DeferredType) or isinstance(false_field, ts.DeferredType):
        return ts.DeferredType(constraint=None)

    @utils.tree_map(
        collection_type=ts.TupleType,
        result_collection_constructor=lambda _, elts: ts.TupleType(types=list(elts)),
    )
    def deduce_return_type(tb: ts.FieldType | ts.ScalarType, fb: ts.FieldType | ts.ScalarType):
        if any(isinstance(b, ts.DeferredType) for b in [tb, fb]):
            return ts.DeferredType(constraint=ts.FieldType)

        tb_dtype, fb_dtype = (type_info.extract_dtype(b) for b in [tb, fb])

        assert tb_dtype == fb_dtype, (
            f"Field arguments must be of same dtype, got '{tb_dtype}' != '{fb_dtype}'."
        )
        dtype = tb_dtype

        return_dims = common.promote_dims(
            domain.dims, type_info.extract_dims(type_info.promote(tb, fb))
        )
        return_type = ts.FieldType(dims=return_dims, dtype=dtype)
        return return_type

    return deduce_return_type(true_field, false_field)


@_register_builtin_type_synthesizer
def broadcast(
    arg: ts.FieldType | ts.ScalarType | ts.DeferredType, dims: tuple[ts.DimensionType]
) -> ts.FieldType | ts.DeferredType:
    if isinstance(arg, ts.DeferredType):
        return arg

    dims_ = [dim.dim for dim in dims]

    if isinstance(arg, ts.FieldType):
        dtype = arg.dtype
    elif isinstance(arg, ts.ScalarType):
        dtype = arg
    else:
        raise AssertionError("`args` need to be either `ts.FieldType` or `ts.ScalarType`.")

    return ts.FieldType(dims=dims_, dtype=dtype)


@_register_builtin_type_synthesizer
def neighbors(
    offset_literal: it_ts.OffsetLiteralType,
    it: it_ts.IteratorType,
    offset_provider_type: common.OffsetProviderType,
) -> ts.ListType:
    assert isinstance(offset_literal, it_ts.OffsetLiteralType) and isinstance(
        offset_literal.value, str
    )
    assert isinstance(it, it_ts.IteratorType)
    conn_type = common.get_offset_type(offset_provider_type, offset_literal.value)
    assert isinstance(conn_type, common.NeighborConnectivityType)
    return ts.ListType(element_type=it.element_type, offset_type=conn_type.neighbor_dim)


@_register_builtin_type_synthesizer
def lift(stencil: TypeSynthesizer) -> TypeSynthesizer:
    @type_synthesizer
    def apply_lift(
        *its: it_ts.IteratorType, offset_provider_type: common.OffsetProviderType
    ) -> it_ts.IteratorType:
        assert all(isinstance(it, it_ts.IteratorType) for it in its)
        stencil_args = [
            it_ts.IteratorType(
                # the positions are only known when we deref
                position_dims="unknown",
                defined_dims=it.defined_dims,
                element_type=it.element_type,
            )
            for it in its
        ]
        stencil_return_type = stencil(*stencil_args, offset_provider_type=offset_provider_type)
        assert isinstance(stencil_return_type, ts.DataType)

        position_dims = its[0].position_dims if its else []
        # we would need to look inside the stencil to find out where the resulting iterator
        # is defined, e.g. using trace shift, instead just use an empty list which means
        # everywhere
        defined_dims: list[common.Dimension] = []
        return it_ts.IteratorType(
            position_dims=position_dims, defined_dims=defined_dims, element_type=stencil_return_type
        )

    return apply_lift


def _collect_and_check_dimensions(input_: ts.TypeSpec) -> list[common.Dimension]:
    """
    Extracts dimensions from non-zero-dimensional field inputs and ensures they match.
    """
    all_input_dims = (
        type_info.primitive_constituents(input_)
        .if_isinstance(ts.FieldType)
        .getattr("dims")
        .filter(lambda dims: len(dims) > 0)
        .to_list()
    )
    return common.promote_dims(*all_input_dims)


def _convert_as_fieldop_input_to_iterator(
    domain: ts.DomainType, input_: ts.TypeSpec
) -> it_ts.IteratorType:
    """
    Convert a field operation input into an iterator type, preserving its dimensions and data type.
    """
    input_dims = _collect_and_check_dimensions(input_)
    element_type: ts.DataType = type_info.apply_to_primitive_constituents(
        type_info.extract_dtype, input_
    )

    return it_ts.IteratorType(
        position_dims=domain.dims, defined_dims=input_dims, element_type=element_type
    )


@overload
def _canonicalize_nb_fields(
    input_: ts.ScalarType | ts.FieldType,
) -> ts.ScalarType | ts.FieldType: ...


@overload
def _canonicalize_nb_fields(
    input_: ts.TupleType | tuple[ts.ScalarType | ts.FieldType | ts.TupleType, ...],
) -> ts.TupleType: ...


def _canonicalize_nb_fields(
    input_: ts.ScalarType
    | ts.FieldType
    | ts.TupleType
    | tuple[ts.ScalarType | ts.FieldType | ts.TupleType, ...],
) -> ts.ScalarType | ts.FieldType | ts.TupleType:
    """
    Transform neighbor / sparse field type by removal of local dimension and addition of corresponding `ListType` dtype.

    Examples:
    >>> input_field = ts.FieldType(
    ...     dims=[
    ...         common.Dimension(value="Vertex"),
    ...         common.Dimension(value="V2E", kind=common.DimensionKind.LOCAL),
    ...     ],
    ...     dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
    ... )
    >>> _canonicalize_nb_fields(input_field)
    FieldType(dims=[Dimension(value='Vertex', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ListType(element_type=ScalarType(kind=<ScalarKind.FLOAT64: 11>, shape=None), offset_type=Dimension(value='V2E', kind=<DimensionKind.LOCAL: 'local'>)))
    """
    match input_:
        case tuple() | ts.TupleType():
            assert all(
                isinstance(field, (ts.ScalarType, ts.FieldType, ts.TupleType)) for field in input_
            )
            return ts.TupleType(
                types=[
                    _canonicalize_nb_fields(cast(ts.FieldType | ts.TupleType, field))
                    for field in input_
                ]
            )
        case ts.FieldType():
            input_dims = _collect_and_check_dimensions(input_)
            element_type: ts.DataType = type_info.apply_to_primitive_constituents(
                type_info.extract_dtype, input_
            )
            defined_dims = []
            neighbor_dim = None
            for dim in input_dims:
                if dim.kind == common.DimensionKind.LOCAL:
                    assert neighbor_dim is None
                    neighbor_dim = dim
                else:
                    defined_dims.append(dim)
            if neighbor_dim:
                element_type = ts.ListType(element_type=element_type, offset_type=neighbor_dim)
            return ts.FieldType(dims=defined_dims, dtype=element_type)
        case ts.ScalarType():
            return input_
        case _:
            raise TypeError(f"Unexpected type: {type(input_)}")


def _resolve_dimensions(
    input_dims: list[common.Dimension],
    shift_tuple: tuple[itir.OffsetLiteral, ...],
    offset_provider_type: common.OffsetProviderType,
) -> list[common.Dimension]:
    """
    Resolves the final dimensions by applying shifts from the given shift tuple.

    Args:
        - input_dims: A list of initial dimensions to resolve.
        - shift_tuple: A tuple of offset literals defining the shift.
        - offset_provider_type: Offset provider dictionary.

    Returns:
        A list of resolved dimensions after applying the shifts.

    Examples:
        Consider the following expression
        ```
        (⇑(λ(it) → ·⟪V2Eₒ, 0ₒ⟫(it)))(inp)
        ```
        where `inp` is a field defined on [Edge, K] that is given to an `as_fieldop` then
        ```
        _resolve_dimensions([Edge, K], (V2Eₒ, 0ₒ), ...)
        ```
        tells you the dimensions of the field returned by the `as_fieldop`, in this case
        `[Vertex, K]`.

        >>> Edge = common.Dimension(value="Edge")
        >>> Vertex = common.Dimension(value="Vertex")
        >>> K = common.Dimension(value="K", kind=common.DimensionKind.VERTICAL)
        >>> V2E = common.Dimension(value="V2E")
        >>> input_dims = [Edge, K]
        >>> shift_tuple = (
        ...     itir.OffsetLiteral(value="V2E"),
        ...     itir.OffsetLiteral(value=0),
        ... )
        >>> offset_provider_type = {
        ...     "V2E": common.NeighborConnectivityType(
        ...         domain=(Vertex, V2E),
        ...         codomain=Edge,
        ...         skip_value=None,
        ...         dtype=None,
        ...         max_neighbors=4,
        ...     ),
        ...     "KOff": K,
        ... }
        >>> _resolve_dimensions(input_dims, shift_tuple, offset_provider_type)
        [Dimension(value='Vertex', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='K', kind=<DimensionKind.VERTICAL: 'vertical'>)]
    """
    resolved_dims = []
    for input_dim in input_dims:
        for off_literal in reversed(
            shift_tuple[::2]
        ):  # Only OffsetLiterals are processed, located at even indices in shift_tuple. Shifts are applied in reverse order: the last shift in the tuple is applied first.
            assert isinstance(off_literal.value, str)
            offset_type = common.get_offset_type(offset_provider_type, off_literal.value)
            if isinstance(offset_type, common.Dimension) and input_dim == offset_type:
                continue  # No shift applied
            if isinstance(offset_type, (fbuiltins.FieldOffset, common.NeighborConnectivityType)):
                if input_dim == offset_type.codomain:  # Check if input fits to offset
                    input_dim = offset_type.domain[0]  # Update input_dim for next iteration
        resolved_dims.append(input_dim)
    return resolved_dims


@_register_builtin_type_synthesizer
def as_fieldop(
    stencil: TypeSynthesizer,
    domain: Optional[ts.DomainType] = None,
    *,
    offset_provider_type: common.OffsetProviderType,
) -> TypeSynthesizer:
    @type_synthesizer
    def applied_as_fieldop(
        *fields: ts.TupleType,
        # For each stencil parameter all locations it is `deref`ed on
        #  see :func:`gt4py.next.iterator.transforms.trace_stencil`.
        shift_sequences_per_param: list[set[tuple[itir.OffsetLiteral, ...]]] | None,
    ) -> ts.FieldType | ts.DeferredType:
        if any(
            isinstance(el, ts.DeferredType)
            for f in fields
            for el in type_info.primitive_constituents(f)
        ):
            return ts.DeferredType(constraint=None)
        nonlocal domain

        new_fields = _canonicalize_nb_fields(fields)

        if not domain:
            deduced_domain = None
            output_dims: list[common.Dimension] = []
            if offset_provider_type is not None and shift_sequences_per_param is not None:
                for field, shift_sequences in zip(
                    new_fields, shift_sequences_per_param, strict=True
                ):
                    for el in type_info.primitive_constituents(field):
                        input_dims = type_info.extract_dims(el)
                        for shift_sequence in shift_sequences:
                            output_dims = common.promote_dims(
                                output_dims,
                                _resolve_dimensions(
                                    input_dims, shift_sequence, offset_provider_type
                                ),
                            )

                        assert all(isinstance(dim, common.Dimension) for dim in output_dims)
                        deduced_domain = ts.DomainType(dims=output_dims)

            if deduced_domain:
                domain = deduced_domain
            else:
                return ts.DeferredType(constraint=None)

        stencil_return = stencil(
            *(_convert_as_fieldop_input_to_iterator(domain, field) for field in new_fields),
            offset_provider_type=offset_provider_type,
        )

        assert isinstance(stencil_return, ts.DataType)

        return type_info.apply_to_primitive_constituents(
            lambda el_type: ts.FieldType(
                dims=domain.dims,
                dtype=el_type,
            ),
            stencil_return,
        )

    return applied_as_fieldop


@_register_builtin_type_synthesizer
def get_domain_range(field: ts.FieldType, dim: ts.DimensionType) -> ts.TupleType:
    return ts.TupleType(
        types=[ts.ScalarType(kind=getattr(ts.ScalarKind, builtins.INTEGER_INDEX_BUILTIN.upper()))]
        * 2
    )


@_register_builtin_type_synthesizer
def scan(
    scan_pass: TypeSynthesizer, direction: ts.ScalarType, init: ts.ScalarType
) -> TypeSynthesizer:
    assert isinstance(direction, ts.ScalarType) and direction.kind == ts.ScalarKind.BOOL

    @type_synthesizer
    def apply_scan(
        *its: it_ts.IteratorType, offset_provider_type: common.OffsetProviderType
    ) -> ts.DataType:
        result = scan_pass(init, *its, offset_provider_type=offset_provider_type)
        assert isinstance(result, ts.DataType)
        return result

    return apply_scan


@_register_builtin_type_synthesizer
def map_(op: TypeSynthesizer) -> TypeSynthesizer:
    @type_synthesizer
    def applied_map(
        *args: ts.ListType, offset_provider_type: common.OffsetProviderType
    ) -> ts.ListType:
        assert len(args) > 0
        assert all(isinstance(arg, ts.ListType) for arg in args)
        arg_el_types = [arg.element_type for arg in args]
        el_type = op(*arg_el_types, offset_provider_type=offset_provider_type)
        assert isinstance(el_type, ts.DataType)
        offset_types = [arg.offset_type for arg in args if arg.offset_type]
        offset_type = offset_types[0] if offset_types else None
        assert all(offset_type == arg for arg in offset_types)
        return ts.ListType(element_type=el_type, offset_type=offset_type)

    return applied_map


@_register_builtin_type_synthesizer
def reduce(op: TypeSynthesizer, init: ts.TypeSpec) -> TypeSynthesizer:
    @type_synthesizer
    def applied_reduce(*args: ts.ListType, offset_provider_type: common.OffsetProviderType):
        assert all(isinstance(arg, ts.ListType) for arg in args)
        return op(
            init, *(arg.element_type for arg in args), offset_provider_type=offset_provider_type
        )

    return applied_reduce


@_register_builtin_type_synthesizer
def shift(*offset_literals, offset_provider_type: common.OffsetProviderType) -> TypeSynthesizer:
    @type_synthesizer
    def apply_shift(
        it: it_ts.IteratorType | ts.DeferredType,
    ) -> it_ts.IteratorType | ts.DeferredType:
        if isinstance(it, ts.DeferredType):
            return ts.DeferredType(constraint=None)
        assert isinstance(it, it_ts.IteratorType)
        if it.position_dims == "unknown":  # nothing to do here
            return it
        new_position_dims: list[common.Dimension] | str
        if offset_provider_type:
            new_position_dims = [*it.position_dims]
            assert len(offset_literals) % 2 == 0
            for offset_axis, _ in zip(offset_literals[:-1:2], offset_literals[1::2], strict=True):
                assert isinstance(offset_axis, it_ts.OffsetLiteralType) and isinstance(
                    offset_axis.value, str
                )
                type_ = common.get_offset_type(offset_provider_type, offset_axis.value)
                if isinstance(type_, common.Dimension):
                    pass
                elif isinstance(type_, common.NeighborConnectivityType):
                    found = False
                    for i, dim in enumerate(new_position_dims):
                        if dim.value == type_.source_dim.value:
                            assert not found
                            new_position_dims[i] = type_.codomain
                            found = True
                    assert found
                else:
                    raise NotImplementedError(f"{type_} is not a supported Connectivity type.")
        else:
            # during re-inference we don't have an offset provider type
            new_position_dims = "unknown"
        return it_ts.IteratorType(
            position_dims=new_position_dims,
            defined_dims=it.defined_dims,
            element_type=it.element_type,
        )

    return apply_shift
