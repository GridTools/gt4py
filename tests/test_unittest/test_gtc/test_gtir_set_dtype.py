from eve.tree_utils import FindNodes

from gt4py.gtc import common
from gt4py.gtc.gtir import FieldAccess, FieldDecl, ParAssignStmt
from gt4py.gtc.passes.gtir_set_dtype import GTIRSetDtype

from .gtir_utils import FieldAccessBuilder, StencilBuilder


A_ARITHMETIC_TYPE = common.DataType.FLOAT32


def test_dtype_FieldAccess():
    name = "foo"
    decl = FieldDecl(name=name, dtype=A_ARITHMETIC_TYPE)

    testee = FieldAccessBuilder(name).build()

    result = GTIRSetDtype().visit(testee, symtable={name: decl})

    assert result.dtype == A_ARITHMETIC_TYPE


def test_stencil():
    field_name = "field"
    testee = (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_par_assign_stmt(
            ParAssignStmt(
                left=FieldAccessBuilder(field_name).build(),
                right=FieldAccessBuilder(field_name).build(),
            )
        )
        .build()
    )

    result = GTIRSetDtype().visit(testee)

    field_accesses = FindNodes.by_type(FieldAccess, result)
    for acc in field_accesses:
        assert acc.dtype == A_ARITHMETIC_TYPE
