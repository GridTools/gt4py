# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import copy

# TODO: test failure when something is not typed after inference is run
# TODO: test lift with no args
# TODO: lambda function that is not called
# TODO: partially applied function in a let
# TODO: function calling itself should fail
# TODO: lambda function called with different argument types

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im, common_pattern_matcher as cpm
from gt4py.next.iterator.type_system import (
    inference as itir_type_inference,
    type_specifications as it_ts,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.cases import (
    C2E,
    E2V,
    V2E,
    E2VDim,
    Edge,
    Cell,
    IDim,
    Ioff,
    JDim,
    KDim,
    Koff,
    V2EDim,
    Vertex,
    exec_alloc_descriptor,
    mesh_descriptor,
    unstructured_case,
)
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import simple_mesh

bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
float64_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
float64_list_type = ts.ListType(element_type=float64_type, offset_type=V2EDim)
int_list_type = ts.ListType(element_type=int_type)

float_i_field = ts.FieldType(dims=[IDim], dtype=float64_type)
float_j_field = ts.FieldType(dims=[JDim], dtype=float64_type)
float_k_field = ts.FieldType(dims=[KDim], dtype=float64_type)
float_ij_field = ts.FieldType(dims=[IDim, JDim], dtype=float64_type)
float_vertex_k_field = ts.FieldType(dims=[Vertex, KDim], dtype=float64_type)
float_edge_k_field = ts.FieldType(dims=[Edge, KDim], dtype=float64_type)
float_edge_field = ts.FieldType(dims=[Edge], dtype=float64_type)
float_vertex_field = ts.FieldType(dims=[Vertex], dtype=float64_type)
float_cell_k_field = ts.FieldType(dims=[Cell, KDim], dtype=float64_type)
float_vertex_v2e_field = ts.FieldType(dims=[Vertex, V2EDim], dtype=float64_type)

it_on_v_of_e_type = it_ts.IteratorType(
    position_dims=[Vertex, KDim], defined_dims=[Edge, KDim], element_type=int_type
)

it_on_e_of_e_type = it_ts.IteratorType(
    position_dims=[Edge, KDim], defined_dims=[Edge, KDim], element_type=int_type
)

it_ijk_type = it_ts.IteratorType(
    position_dims=[IDim, JDim, KDim], defined_dims=[IDim, JDim, KDim], element_type=int_type
)


def expression_test_cases():
    i_domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)})
    ij_domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 1), JDim: (0, 1)})
    k_domain = im.domain(common.GridType.CARTESIAN, {KDim: (0, 1)})
    cell_k_domain = im.domain(common.GridType.UNSTRUCTURED, {Cell: (0, 1), KDim: (0, 1)})
    vertex_k_domain = im.domain(common.GridType.UNSTRUCTURED, {Vertex: (0, 1), KDim: (0, 1)})

    return (
        # itir expr, type
        (im.call("abs")(1), int_type),
        (im.call("power")(2.0, 2), float64_type),
        (im.plus(1, 2), int_type),
        (im.eq(1, 2), bool_type),
        (im.deref(im.ref("it", it_on_e_of_e_type)), it_on_e_of_e_type.element_type),
        (im.can_deref(im.ref("it", it_on_e_of_e_type)), bool_type),
        (im.if_(True, 1, 2), int_type),
        (im.call("make_const_list")(True), ts.ListType(element_type=bool_type)),
        (im.list_get(0, im.ref("l", ts.ListType(element_type=bool_type))), bool_type),
        (
            im.call("named_range")(
                itir.AxisLiteral(value="Vertex", kind=common.DimensionKind.HORIZONTAL), 0, 1
            ),
            it_ts.NamedRangeType(dim=Vertex),
        ),
        (
            im.call("cartesian_domain")(
                im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
            ),
            ts.DomainType(dims=[IDim]),
        ),
        (
            im.call("unstructured_domain")(
                im.call("named_range")(
                    itir.AxisLiteral(value="Vertex", kind=common.DimensionKind.HORIZONTAL), 0, 1
                )
            ),
            ts.DomainType(dims=[Vertex]),
        ),
        # make_tuple
        (
            im.make_tuple(im.ref("a", int_type), im.ref("b", bool_type)),
            ts.TupleType(types=[int_type, bool_type]),
        ),
        # tuple_get
        (im.tuple_get(0, im.make_tuple(im.ref("a", int_type), im.ref("b", bool_type))), int_type),
        (im.tuple_get(1, im.make_tuple(im.ref("a", int_type), im.ref("b", bool_type))), bool_type),
        (
            im.tuple_get(0, im.ref("t", ts.DeferredType(constraint=None))),
            ts.DeferredType(constraint=None),
        ),
        # neighbors
        (
            im.neighbors("E2V", im.ref("a", it_on_e_of_e_type)),
            ts.ListType(element_type=it_on_e_of_e_type.element_type, offset_type=E2VDim),
        ),
        # cast
        (im.cast_(1, int_type), int_type),
        # TODO: lift
        # TODO: scan
        # map
        (
            im.map_(im.ref("plus"))(im.ref("a", int_list_type), im.ref("b", int_list_type)),
            int_list_type,
        ),
        (
            im.map_(im.ref("plus"))(
                im.ref("a", int_list_type),
                im.ref("b", ts.ListType(element_type=int_type, offset_type=V2EDim)),
            ),
            ts.ListType(element_type=int_type, offset_type=V2EDim),
        ),
        # reduce
        (im.reduce("plus", 0)(im.ref("l", int_list_type)), int_type),
        (
            im.reduce(
                im.lambda_("acc", "a", "b")(
                    im.make_tuple(
                        im.plus(im.tuple_get(0, "acc"), "a"),
                        im.plus(im.tuple_get(1, "acc"), "b"),
                    )
                ),
                im.make_tuple(0, 0.0),
            )(im.ref("la", int_list_type), im.ref("lb", float64_list_type)),
            ts.TupleType(types=[int_type, float64_type]),
        ),
        # shift
        (im.shift("V2E", 1)(im.ref("it", it_on_v_of_e_type)), it_on_e_of_e_type),
        (im.shift("Ioff", 1)(im.ref("it", it_ijk_type)), it_ijk_type),
        # as_fieldop
        (
            im.as_fieldop("deref", i_domain)(im.ref("inp", float_i_field)),
            float_i_field,
        ),
        (
            im.as_fieldop(
                im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))),
                i_domain,
            )(im.ref("inp", float_i_field)),
            float_i_field,
        ),
        (
            im.as_fieldop(
                im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))),
                ij_domain,
            )(im.ref("inp", float_ij_field)),
            float_ij_field,
        ),
        (
            im.as_fieldop(
                im.lambda_("it")(im.deref(im.shift("Koff", 1)(im.shift("V2E", 0)("it")))),
                vertex_k_domain,
            )(im.ref("inp", float_edge_k_field)),
            float_vertex_k_field,
        ),
        (
            im.as_fieldop(
                im.lambda_("it1", "it2")(
                    im.plus(
                        im.deref(im.shift("E2V", 1)(im.shift("C2E", 1)("it1"))),
                        im.deref(im.shift("Koff", 1)("it2")),
                    ),
                ),
                cell_k_domain,
            )(
                im.ref("inp1", float_vertex_field),
                im.ref("inp2", float_k_field),
            ),
            float_cell_k_field,
        ),
        (
            im.as_fieldop(
                im.lambda_("it")(im.deref(im.shift("V2E", 0)("it"))),
                vertex_k_domain,
            )(im.ref("inp", float_edge_k_field)),
            float_vertex_k_field,
        ),
        (
            im.as_fieldop(
                im.lambda_("a", "b")(im.make_tuple(im.deref("a"), im.deref("b"))),
                ij_domain,
            )(im.ref("inp1", float_i_field), im.ref("inp2", float_j_field)),
            ts.TupleType(types=[float_ij_field, float_ij_field]),
        ),
        (
            im.as_fieldop(
                im.scan(im.lambda_("state", "inp")(im.plus("state", im.deref("inp"))), True, 0.0),
                k_domain,
            )(im.ref("inp", float_k_field)),
            float_k_field,
        ),
        (
            im.as_fieldop(im.lambda_("x")(im.deref("x")))(
                im.ref("inp", ts.DeferredType(constraint=None))
            ),
            ts.DeferredType(constraint=None),
        ),
        # if in field-view scope
        (
            im.if_(
                False,
                im.as_fieldop(
                    im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                    im.call("cartesian_domain")(
                        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
                    ),
                )(im.ref("inp", float_i_field), 1.0),
                im.as_fieldop(
                    "deref",
                    im.call("cartesian_domain")(
                        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
                    ),
                )(im.ref("inp", float_i_field)),
            ),
            float_i_field,
        ),
        (
            im.if_(
                False,
                im.make_tuple(im.ref("inp", float_i_field), im.ref("inp", float_i_field)),
                im.make_tuple(im.ref("inp", float_i_field), im.ref("inp", float_i_field)),
            ),
            ts.TupleType(types=[float_i_field, float_i_field]),
        ),
        # concat_where
        (
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)}),
                im.ref("a", float_i_field),
                im.ref("b", float_ij_field),
            ),
            float_ij_field,
        ),
        (
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)}),
                im.ref("a", ts.TupleType(types=[float_i_field] * 2)),
                im.ref("b", ts.TupleType(types=[float_i_field] * 2)),
            ),
            ts.TupleType(types=[float_i_field] * 2),
        ),
        (
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)}),
                im.ref("a", ts.TupleType(types=[float_i_field, float_ij_field])),
                im.ref("b", ts.TupleType(types=[float_i_field] * 2)),
            ),
            ts.TupleType(types=[float_i_field, float_ij_field]),
        ),
    )


@pytest.mark.parametrize("test_case", expression_test_cases())
def test_expression_type(test_case):
    mesh = simple_mesh(None)
    offset_provider_type = {**mesh.offset_provider_type, "Ioff": IDim, "Joff": JDim, "Koff": KDim}

    testee, expected_type = test_case
    result = itir_type_inference.infer(
        testee, offset_provider_type=offset_provider_type, allow_undeclared_symbols=True
    )
    assert result.type == expected_type


@pytest.mark.parametrize(
    "test_case",
    [(expr, type_) for expr, type_ in expression_test_cases() if cpm.is_applied_as_fieldop(expr)],
)
def test_expression_type_as_fieldop_no_domain(test_case):
    mesh = simple_mesh(None)
    offset_provider_type = {**mesh.offset_provider_type, "Ioff": IDim, "Joff": JDim, "Koff": KDim}

    testee_with_domain, expected_type = test_case
    result_with_domain = itir_type_inference.infer(
        testee_with_domain, offset_provider_type=offset_provider_type, allow_undeclared_symbols=True
    )
    # testee stays as is, but we remove the domain
    testee_without_domain = im.as_fieldop(testee_with_domain.fun.args[0])(*testee_with_domain.args)
    result_without_domain = itir_type_inference.infer(
        testee_without_domain,
        offset_provider_type=offset_provider_type,
        allow_undeclared_symbols=True,
    )
    assert result_with_domain.type == result_without_domain.type == expected_type


def test_adhoc_polymorphism():
    func = im.lambda_("a")(im.lambda_("b")(im.make_tuple("a", "b")))
    testee = im.call(im.call(func)(im.ref("a_", bool_type)))(im.ref("b_", int_type))

    result = itir_type_inference.infer(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )

    assert result.type == ts.TupleType(types=[bool_type, int_type])


def test_binary_lambda():
    func = im.lambda_("a", "b")(im.make_tuple("a", "b"))
    testee = im.call(func)(im.ref("a_", bool_type), im.ref("b_", int_type))

    result = itir_type_inference.infer(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )

    expected_type = ts.TupleType(types=[bool_type, int_type])
    assert result.type == expected_type
    assert result.fun.params[0].type == bool_type
    assert result.fun.params[1].type == int_type
    assert result.fun.type == ts.FunctionType(
        pos_only_args=[bool_type, int_type],
        pos_or_kw_args={},
        kw_only_args={},
        returns=expected_type,
    )


def test_aliased_function():
    testee = im.let("f", im.lambda_("x")("x"))(im.call("f")(1))
    result = itir_type_inference.infer(testee, offset_provider_type={})

    assert result.args[0].type == ts.FunctionType(
        pos_only_args=[int_type], pos_or_kw_args={}, kw_only_args={}, returns=int_type
    )
    assert result.args[0].params[0].type == int_type
    assert result.type == int_type


def test_late_offset_axis():
    mesh = simple_mesh(None)

    func = im.lambda_("dim")(im.shift(im.ref("dim"), 1)(im.ref("it", it_on_v_of_e_type)))
    testee = im.call(func)(im.ensure_offset("V2E"))

    result = itir_type_inference.infer(
        testee, offset_provider_type=mesh.offset_provider_type, allow_undeclared_symbols=True
    )
    assert result.type == it_on_e_of_e_type


def test_cast_first_arg_inference():
    # since cast_ is a grammar builtin whose return type is given by its second argument it is
    # easy to forget inferring the types of the first argument and its children. Simply check
    # if the first argument has a type inferred correctly here.
    testee = im.cast_(im.plus(im.literal_from_value(1), im.literal_from_value(2)), "float64")
    result = itir_type_inference.infer(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )

    assert result.args[0].type == int_type
    assert result.type == float64_type


def test_cartesian_fencil_definition():
    cartesian_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
    )

    testee = itir.Program(
        id="f",
        function_definitions=[],
        params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"), cartesian_domain)(im.ref("inp")),
                domain=cartesian_domain,
                target=im.ref("out"),
            ),
        ],
    )

    result = itir_type_inference.infer(testee, offset_provider_type={"Ioff": IDim})

    program_type = it_ts.ProgramType(params={"inp": float_i_field, "out": float_i_field})
    assert result.type == program_type
    domain_type = ts.DomainType(dims=[IDim])
    assert result.body[0].domain.type == domain_type
    assert result.body[0].expr.type == float_i_field
    assert result.body[0].target.type == float_i_field


def test_unstructured_fencil_definition():
    mesh = simple_mesh(None)
    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(
            itir.AxisLiteral(value="Vertex", kind=common.DimensionKind.HORIZONTAL), 0, 1
        ),
        im.call("named_range")(
            itir.AxisLiteral(value="KDim", kind=common.DimensionKind.VERTICAL), 0, 1
        ),
    )

    testee = itir.Program(
        id="f",
        function_definitions=[],
        params=[im.sym("inp", float_edge_k_field), im.sym("out", float_vertex_k_field)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(im.deref(im.shift("V2E", 0)("it"))), unstructured_domain
                )(im.ref("inp")),
                domain=unstructured_domain,
                target=im.ref("out"),
            ),
        ],
    )

    result = itir_type_inference.infer(testee, offset_provider_type=mesh.offset_provider_type)

    program_type = it_ts.ProgramType(
        params={"inp": float_edge_k_field, "out": float_vertex_k_field}
    )
    assert result.type == program_type
    domain_type = ts.DomainType(dims=[Vertex, KDim])
    assert result.body[0].domain.type == domain_type
    assert result.body[0].expr.type == float_vertex_k_field
    assert result.body[0].target.type == float_vertex_k_field


def test_function_definition():
    cartesian_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
    )

    testee = itir.Program(
        id="f",
        function_definitions=[
            itir.FunctionDefinition(id="foo", params=[im.sym("it")], expr=im.deref("it")),
            itir.FunctionDefinition(id="bar", params=[im.sym("it")], expr=im.call("foo")("it")),
        ],
        params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
        declarations=[],
        body=[
            itir.SetAt(
                domain=cartesian_domain,
                expr=im.as_fieldop(im.ref("bar"), cartesian_domain)(im.ref("inp")),
                target=im.ref("out"),
            ),
        ],
    )

    result = itir_type_inference.infer(testee, offset_provider_type={"Ioff": IDim})

    program_type = it_ts.ProgramType(params={"inp": float_i_field, "out": float_i_field})
    assert result.type == program_type
    assert result.body[0].expr.type == float_i_field
    assert result.body[0].target.type == float_i_field


def test_fencil_with_nb_field_input():
    mesh = simple_mesh(None)
    unstructured_domain = im.call("unstructured_domain")(
        im.call("named_range")(
            itir.AxisLiteral(value="Vertex", kind=common.DimensionKind.HORIZONTAL), 0, 1
        ),
        im.call("named_range")(
            itir.AxisLiteral(value="KDim", kind=common.DimensionKind.VERTICAL), 0, 1
        ),
    )

    testee = itir.Program(
        id="f",
        function_definitions=[],
        params=[im.sym("inp", float_vertex_v2e_field), im.sym("out", float_vertex_k_field)],
        declarations=[],
        body=[
            itir.SetAt(
                domain=unstructured_domain,
                expr=im.as_fieldop(
                    im.lambda_("it")(im.reduce("plus", 0.0)(im.deref("it"))),
                    unstructured_domain,
                )(im.ref("inp")),
                target=im.ref("out"),
            ),
        ],
    )

    result = itir_type_inference.infer(testee, offset_provider_type=mesh.offset_provider_type)
    stencil = result.body[0].expr.fun.args[0]
    assert stencil.expr.args[0].type == float64_list_type
    assert stencil.type.returns == float64_type


def test_program_tuple_setat_short_target():
    cartesian_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
    )

    testee = itir.Program(
        id="f",
        function_definitions=[],
        params=[im.sym("out", float_i_field)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_()(im.make_tuple(1.0, 2.0)), cartesian_domain)(),
                domain=cartesian_domain,
                target=im.make_tuple("out"),
            )
        ],
    )

    result = itir_type_inference.infer(testee, offset_provider_type={"Ioff": IDim})

    assert (
        isinstance(result.body[0].expr.type, ts.TupleType)
        and len(result.body[0].expr.type.types) == 2
    )
    assert (
        isinstance(result.body[0].target.type, ts.TupleType)
        and len(result.body[0].target.type.types) == 1
    )


def test_program_setat_without_domain():
    cartesian_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
    )

    testee = itir.Program(
        id="f",
        function_definitions=[],
        params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_("x")(im.deref("x")))("inp"),
                domain=cartesian_domain,
                target=im.ref("out", float_i_field),
            )
        ],
    )

    result = itir_type_inference.infer(testee, offset_provider_type={"Ioff": IDim})

    assert result.body[0].expr.type, ts.FieldType(dims=[IDim], dtype=float64_type)


def test_if_stmt():
    cartesian_domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 1)
    )

    testee = itir.IfStmt(
        cond=im.literal_from_value(True),
        true_branch=[
            itir.SetAt(
                expr=im.as_fieldop("deref", cartesian_domain)(im.ref("inp", float_i_field)),
                domain=cartesian_domain,
                target=im.ref("out", float_i_field),
            )
        ],
        false_branch=[],
    )

    result = itir_type_inference.infer(
        testee, offset_provider_type={}, allow_undeclared_symbols=True
    )
    assert result.cond.type == bool_type
    assert result.true_branch[0].expr.type == float_i_field


def test_as_fieldop_without_domain_nb_field_input():
    stencil = im.lambda_("it1")(im.deref("it1"))

    testee = im.as_fieldop(stencil)(im.ref("inp1", float_vertex_v2e_field))
    result = itir_type_inference.infer(
        testee, offset_provider_type={"V2E": V2E}, allow_undeclared_symbols=True
    )
    assert result.type == ts.FieldType(dims=[Vertex], dtype=float64_list_type)
    assert result.fun.args[0].type.pos_only_args[0] == it_ts.IteratorType(
        position_dims=[Vertex],
        defined_dims=float_vertex_field.dims,
        element_type=float64_list_type,
    )


def test_reinference():
    testee = im.make_tuple(im.ref("inp1", float_i_field), im.ref("inp2", float_i_field))
    result = itir_type_inference.reinfer(copy.deepcopy(testee))
    assert result.type == ts.TupleType(types=[float_i_field, float_i_field])


def test_func_reinference():
    f_type = ts.FunctionType(
        pos_only_args=[],
        pos_or_kw_args={},
        kw_only_args={},
        returns=float_i_field,
    )
    testee = im.call(im.ref("f", f_type))()
    result = itir_type_inference.reinfer(copy.deepcopy(testee))
    assert result.type == float_i_field
