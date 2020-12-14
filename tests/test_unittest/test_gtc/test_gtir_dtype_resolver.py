from typing import Dict, Tuple

from gt4py.gtc import common
from gt4py.gtc.common import DataType
from gt4py.gtc.gtir import FieldDecl, ParAssignStmt, Stencil, Literal
from gt4py.gtc.passes.gtir_dtype_resolver import _GTIRPropagateDtypeToAccess, resolve_dtype
import pytest

from .gtir_utils import FieldAccessBuilder, StencilBuilder, VerticalLoopBuilder


A_ARITHMETIC_TYPE = common.DataType.FLOAT32


def test_propagate_dtype_to_FieldAccess():
    name = "foo"
    decl = FieldDecl(name=name, dtype=A_ARITHMETIC_TYPE)

    testee = FieldAccessBuilder(name).build()

    result = _GTIRPropagateDtypeToAccess().visit(testee, symtable={name: decl})

    assert result.dtype == A_ARITHMETIC_TYPE


def get_nodes_with_name(stencil: Stencil, name: str):
    return stencil.iter_tree().if_hasattr("name").filter(lambda node: node.name == name).to_list()


def propagate_dtype_to_FieldAccess():
    return (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_par_assign_stmt(
            ParAssignStmt(
                left=FieldAccessBuilder("field").dtype(None).build(),
                right=FieldAccessBuilder("field").dtype(None).build(),
            )
        )
        .build(),
        {"field": A_ARITHMETIC_TYPE},
    )


def resolve_AUTO_from_literal_to_temporary():
    return (
        StencilBuilder()
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_temporary("tmp", DataType.AUTO)
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp").dtype(None).build(),
                    right=Literal(value="0", dtype=A_ARITHMETIC_TYPE),
                )
            )
            .build()
        )
        .build(),
        {"tmp": A_ARITHMETIC_TYPE},
    )


def resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary():
    return (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_temporary("tmp", DataType.AUTO)
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp").dtype(None).build(),
                    right=FieldAccessBuilder("field").dtype(None).build(),
                )
            )
            .build()
        )
        .build(),
        {"field": A_ARITHMETIC_TYPE, "tmp": A_ARITHMETIC_TYPE},
    )


def resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary_to_FieldAccess_to_temporary():
    return (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_temporary("tmp1", DataType.AUTO)
            .add_temporary("tmp2", DataType.AUTO)
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp1").dtype(None).build(),
                    right=FieldAccessBuilder("field").dtype(None).build(),
                )
            )
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp2").dtype(None).build(),
                    right=FieldAccessBuilder("tmp1").dtype(None).build(),
                ),
            )
            .build()
        )
        .build(),
        {"field": A_ARITHMETIC_TYPE, "tmp1": A_ARITHMETIC_TYPE, "tmp2": A_ARITHMETIC_TYPE},
    )


@pytest.fixture(
    params=[
        propagate_dtype_to_FieldAccess,
        resolve_AUTO_from_literal_to_temporary,
        resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary,
        resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary_to_FieldAccess_to_temporary,
    ]
)
def resolve_dtype_stencil_and_expected_dtypes(request):
    return request.param()


def test_resolved_dtypes(
    resolve_dtype_stencil_and_expected_dtypes: Tuple[Stencil, Dict[str, common.DataType]]
):
    stencil, expected_dtypes = resolve_dtype_stencil_and_expected_dtypes
    # ensure consistency (input is not already fully resolved)
    for name, dtype in expected_dtypes.items():
        nodes = get_nodes_with_name(stencil, name)
        assert len(nodes) > 0
        assert any([node.dtype is None for node in nodes])

    result: Stencil = resolve_dtype(stencil)

    for name, dtype in expected_dtypes.items():
        nodes = get_nodes_with_name(result, name)
        assert len(nodes) > 0
        assert all([node.dtype == dtype for node in nodes])
