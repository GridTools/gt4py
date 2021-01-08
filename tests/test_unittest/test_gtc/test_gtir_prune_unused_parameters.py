from gt4py.gtc import common
from gt4py.gtc.gtir import FieldDecl, ScalarDecl
from gt4py.gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters

from .gtir_utils import (
    ParAssignStmtBuilder,
    StencilBuilder,
)

from devtools import debug

A_ARITHMETIC_TYPE = common.DataType.FLOAT32


def test_all_parameters_used():
    field_param = FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE)
    scalar_param = ScalarDecl(name="scalar", dtype=A_ARITHMETIC_TYPE)
    testee = (
        StencilBuilder()
        .add_param(field_param)
        .add_param(scalar_param)
        .add_par_assign_stmt(ParAssignStmtBuilder("field", "scalar").build())
        .build()
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params


def test_unused_are_removed():
    field_param = FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE)
    unused_field_param = FieldDecl(name="unused_field", dtype=A_ARITHMETIC_TYPE)
    scalar_param = ScalarDecl(name="scalar", dtype=A_ARITHMETIC_TYPE)
    unused_scalar_param = ScalarDecl(name="unused_scalar", dtype=A_ARITHMETIC_TYPE)
    testee = (
        StencilBuilder()
        .add_param(field_param)
        .add_param(unused_field_param)
        .add_param(scalar_param)
        .add_param(unused_scalar_param)
        .add_par_assign_stmt(ParAssignStmtBuilder("field", "scalar").build())
        .build()
    )
    expected_params = [field_param, scalar_param]

    result = prune_unused_parameters(testee)

    assert expected_params == result.params
