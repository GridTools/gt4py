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


from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import Iterable, Optional, Union
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.type_system import type_specifications as it_ts
from gt4py.next.type_system import type_info, type_specifications as ts


TypeSpecOrTypeInferenceRule = Union[ts.TypeSpec, "TypeInferenceRule"]

TypeInferenceRule = xtyping.Callable[..., TypeSpecOrTypeInferenceRule]

#: dictionary from function name to its type inference rule
type_inference_rules: dict[str, TypeInferenceRule] = {}


def _is_derefable_iterator_type(it_type: it_ts.IteratorType) -> bool:
    if it_type.position_dims == "unknown":
        return True
    it_position_dim_names = [dim.value for dim in it_type.position_dims]  # TODO
    return all(dim.value in it_position_dim_names for dim in it_type.defined_dims)


def _register_type_inference_rule(
    rule: Optional[TypeInferenceRule] = None, *, fun_names: Optional[Iterable[str]] = None
):
    def wrapper(rule):
        nonlocal fun_names
        if not fun_names:
            fun_names = [rule.__name__]
        else:
            # store names in function object for better debuggability
            rule.fun_names = fun_names
        for fun_ in fun_names:
            type_inference_rules[fun_] = rule

    if rule:
        return wrapper(rule)
    else:
        return wrapper


@_register_type_inference_rule(
    fun_names=itir.UNARY_MATH_NUMBER_BUILTINS | itir.UNARY_MATH_FP_BUILTINS
)
def _(val: ts.ScalarType) -> ts.ScalarType:
    return val


@_register_type_inference_rule
def power(base: ts.ScalarType, exponent: ts.ScalarType) -> ts.ScalarType:
    return base


@_register_type_inference_rule(fun_names=itir.BINARY_MATH_NUMBER_BUILTINS)
def _(lhs: ts.ScalarType, rhs: ts.ScalarType) -> ts.ScalarType:
    assert lhs == rhs
    return lhs


@_register_type_inference_rule(
    fun_names=itir.UNARY_MATH_FP_PREDICATE_BUILTINS | itir.UNARY_LOGICAL_BUILTINS
)
def _(arg: ts.ScalarType) -> ts.ScalarType:
    return ts.ScalarType(kind=ts.ScalarKind.BOOL)


@_register_type_inference_rule(
    fun_names=itir.BINARY_MATH_COMPARISON_BUILTINS | itir.BINARY_LOGICAL_BUILTINS
)
def _(lhs: ts.ScalarType, rhs: ts.ScalarType) -> ts.ScalarType | ts.TupleType:
    return ts.ScalarType(kind=ts.ScalarKind.BOOL)


@_register_type_inference_rule
def deref(it: it_ts.IteratorType) -> ts.DataType:
    assert isinstance(it, it_ts.IteratorType)
    assert _is_derefable_iterator_type(it)
    return it.element_type


@_register_type_inference_rule
def can_deref(it: it_ts.IteratorType) -> ts.ScalarType:
    assert isinstance(it, it_ts.IteratorType)
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


@_register_type_inference_rule
def if_(cond: ts.ScalarType, true_branch: ts.DataType, false_branch: ts.DataType) -> ts.DataType:
    assert isinstance(cond, ts.ScalarType) and cond.kind == ts.ScalarKind.BOOL
    # TODO(tehrengruber): Enable this or a similar check. In case the true- and false-branch are
    #  iterators defined on different positions this fails. For the GTFN backend we also don't
    #  want this, but for roundtrip it is totally fine.
    # assert true_branch == false_branch  # noqa: ERA001
    return true_branch


@_register_type_inference_rule
def make_const_list(scalar: ts.ScalarType) -> it_ts.ListType:
    assert isinstance(scalar, ts.ScalarType)
    return it_ts.ListType(element_type=scalar)


@_register_type_inference_rule
def list_get(index: ts.ScalarType | it_ts.OffsetLiteralType, list_: it_ts.ListType) -> ts.DataType:
    if isinstance(index, it_ts.OffsetLiteralType):
        assert isinstance(index.value, ts.ScalarType)
        index = index.value
    assert isinstance(index, ts.ScalarType) and type_info.is_integral(index)
    assert isinstance(list_, it_ts.ListType)
    return list_.element_type


@_register_type_inference_rule
def named_range(
    dim: ts.DimensionType, start: ts.ScalarType, stop: ts.ScalarType
) -> it_ts.NamedRangeType:
    assert isinstance(dim, ts.DimensionType)
    return it_ts.NamedRangeType(dim=dim.dim)


@_register_type_inference_rule(fun_names=["cartesian_domain", "unstructured_domain"])
def _(*args: it_ts.NamedRangeType) -> it_ts.DomainType:
    assert all(isinstance(arg, it_ts.NamedRangeType) for arg in args)
    return it_ts.DomainType(dims=[arg.dim for arg in args])


@_register_type_inference_rule
def make_tuple(*args: ts.DataType) -> ts.TupleType:
    return ts.TupleType(types=list(args))


@_register_type_inference_rule
def neighbors(offset_literal: it_ts.OffsetLiteralType, it: it_ts.IteratorType) -> it_ts.ListType:
    assert (
        isinstance(offset_literal, it_ts.OffsetLiteralType)
        and isinstance(offset_literal.value, common.Dimension)
        and offset_literal.value.kind == common.DimensionKind.LOCAL
    )
    assert isinstance(it, it_ts.IteratorType)
    return it_ts.ListType(element_type=it.element_type)


@_register_type_inference_rule
def lift(stencil: TypeInferenceRule) -> TypeInferenceRule:
    def apply_lift(*its: it_ts.IteratorType) -> it_ts.IteratorType:
        stencil_args = []
        for it in its:
            assert isinstance(it, it_ts.IteratorType)
            stencil_args.append(
                it_ts.IteratorType(
                    # the positions are only known when we deref
                    position_dims="unknown",
                    defined_dims=it.defined_dims,
                    element_type=it.element_type,
                )
            )
        stencil_return_type = stencil(*stencil_args)
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


def _convert_as_fieldop_input_to_iterator(
    domain: it_ts.DomainType, input_: ts.TypeSpec
) -> it_ts.IteratorType:
    # get the dimensions of all non-zero-dimensional field inputs and check they agree
    all_input_dims = (
        type_info.primitive_constituents(input_)
        .if_isinstance(ts.FieldType)
        .getattr("dims")
        .filter(lambda dims: len(dims) > 0)
        .to_list()
    )
    input_dims: list[common.Dimension]
    if all_input_dims:
        assert all(cur_input_dims == all_input_dims[0] for cur_input_dims in all_input_dims)
        input_dims = all_input_dims[0]
    else:
        input_dims = []

    element_type: ts.DataType
    element_type = type_info.apply_to_primitive_constituents(type_info.extract_dtype, input_)

    # handle neighbor / sparse input fields
    defined_dims = []
    is_nb_field = False
    for dim in input_dims:
        if dim.kind == common.DimensionKind.LOCAL:
            assert not is_nb_field
            is_nb_field = True
        else:
            defined_dims.append(dim)
    if is_nb_field:
        element_type = it_ts.ListType(element_type=element_type)

    return it_ts.IteratorType(
        position_dims=domain.dims, defined_dims=defined_dims, element_type=element_type
    )


@_register_type_inference_rule
def as_fieldop(stencil: TypeInferenceRule, domain: it_ts.DomainType) -> TypeInferenceRule:
    def applied_as_fieldop(*fields) -> ts.FieldType:
        stencil_return = stencil(
            *(_convert_as_fieldop_input_to_iterator(domain, field) for field in fields)
        )
        assert isinstance(stencil_return, ts.DataType)
        return type_info.apply_to_primitive_constituents(
            lambda el_type: ts.FieldType(dims=domain.dims, dtype=el_type), stencil_return
        )

    return applied_as_fieldop


@_register_type_inference_rule
def scan(
    scan_pass: TypeInferenceRule, direction: ts.ScalarType, init: ts.ScalarType
) -> TypeInferenceRule:
    assert isinstance(direction, ts.ScalarType) and direction.kind == ts.ScalarKind.BOOL

    def apply_scan(*its: it_ts.IteratorType) -> ts.DataType:
        result = scan_pass(init, *its)
        assert isinstance(result, ts.DataType)
        return result

    return apply_scan


@_register_type_inference_rule
def map_(op: TypeInferenceRule) -> TypeInferenceRule:
    def applied_map(*args: it_ts.ListType) -> it_ts.ListType:
        assert len(args) > 0
        assert all(isinstance(arg, it_ts.ListType) for arg in args)
        arg_el_types = [arg.element_type for arg in args]
        el_type = op(*arg_el_types)
        assert isinstance(el_type, ts.DataType)
        return it_ts.ListType(element_type=el_type)

    return applied_map


@_register_type_inference_rule
def reduce(op: TypeInferenceRule, init: ts.TypeSpec) -> TypeInferenceRule:
    def applied_reduce(*args: it_ts.ListType):
        assert all(isinstance(arg, it_ts.ListType) for arg in args)
        return op(init, *(arg.element_type for arg in args))

    return applied_reduce


@_register_type_inference_rule
def shift(*offset_literals, offset_provider) -> TypeInferenceRule:
    def apply_shift(it: it_ts.IteratorType) -> it_ts.IteratorType:
        assert isinstance(it, it_ts.IteratorType)
        if it.position_dims == "unknown":  # nothing to do here
            return it
        new_position_dims = [*it.position_dims]
        assert len(offset_literals) % 2 == 0
        for offset_axis, _ in zip(offset_literals[:-1:2], offset_literals[1::2], strict=True):
            assert isinstance(offset_axis, it_ts.OffsetLiteralType) and isinstance(
                offset_axis.value, common.Dimension
            )
            provider = offset_provider[offset_axis.value.value]  # TODO: naming
            if isinstance(provider, common.Dimension):
                pass
            elif isinstance(provider, common.Connectivity):
                found = False
                for i, dim in enumerate(new_position_dims):
                    if dim.value == provider.origin_axis.value:
                        assert not found
                        new_position_dims[i] = provider.neighbor_axis
                        found = True
                assert found
            else:
                raise NotImplementedError()
        return it_ts.IteratorType(
            position_dims=new_position_dims,
            defined_dims=it.defined_dims,
            element_type=it.element_type,
        )

    return apply_shift
