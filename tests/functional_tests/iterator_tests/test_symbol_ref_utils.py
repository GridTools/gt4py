from dataclasses import dataclass
from typing import Optional

import eve
from functional.iterator import ir as itir
from functional.iterator.transforms.symbol_ref_utils import (
    collect_symbol_refs,
    get_user_defined_symbols,
)


def test_get_user_defined_symbols():
    @dataclass
    class _GetRealItirSymTable(eve.VisitorWithSymbolTableTrait):
        extracted_symtable: Optional[dict] = None

        @classmethod
        def apply(cls, ir: itir.Node):
            obj = cls()
            obj.visit(ir)
            return obj.extracted_symtable

        def visit_SymRef(self, node: itir.SymRef, **kwargs):
            self.extracted_symtable = kwargs["symtable"]

    ir = itir.FencilDefinition(
        id="foo",
        function_definitions=[],
        params=[itir.Sym(id="target_symbol")],
        closures=[
            itir.StencilClosure(
                domain=itir.FunCall(fun=itir.SymRef(id="cartesian_domain"), args=[]),
                stencil=itir.SymRef(id="deref"),
                output=itir.SymRef(id="target_symbol"),
                inputs=[],
            )
        ],
    )
    testee = _GetRealItirSymTable.apply(ir)
    actual = get_user_defined_symbols(testee)
    assert actual == {"target_symbol"}


def test_collect_symbol_refs():
    ...
    # TODO
