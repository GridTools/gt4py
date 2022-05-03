from functional.iterator import ir
from functional.iterator.transforms.extract_function import extract_function


def test_lambda():
    testee = ir.Lambda(
        params=[ir.Sym(id="inp")],
        expr=ir.SymRef(id="inp"),
    )
    expected_fundef = ir.FunctionDefinition(
        id="foo",
        params=[ir.Sym(id="inp")],
        expr=ir.SymRef(id="inp"),
    )
    expected_ref = ir.SymRef(id="foo")

    ref, fundef = extract_function(testee, "foo")

    assert fundef == expected_fundef
    assert ref == expected_ref
