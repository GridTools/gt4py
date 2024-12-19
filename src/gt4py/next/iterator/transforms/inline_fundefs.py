# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import symbol_ref_utils


class InlineFundefs(PreserveLocationVisitor, NodeTranslator):
    def visit_SymRef(self, node: itir.SymRef, *, symtable: Dict[str, Any]):
        if node.id in symtable and isinstance(
            (symbol := symtable[node.id]), itir.FunctionDefinition
        ):
            return itir.Lambda(
                params=self.generic_visit(symbol.params, symtable=symtable),
                expr=self.generic_visit(symbol.expr, symtable=symtable),
            )
        return self.generic_visit(node)

    def visit_Program(self, node: itir.Program):
        return self.generic_visit(node, symtable=node.annex.symtable)


def prune_unreferenced_fundefs(program: itir.Program) -> itir.Program:
    """
    Remove all function declarations that are never called.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> fun1 = itir.FunctionDefinition(
    ...     id="fun1",
    ...     params=[im.sym("a")],
    ...     expr=im.call("deref")("a"),
    ... )
    >>> fun2 = itir.FunctionDefinition(
    ...     id="fun2",
    ...     params=[im.sym("a")],
    ...     expr=im.call("deref")("a"),
    ... )
    >>> program = itir.Program(
    ...     id="testee",
    ...     function_definitions=[fun1, fun2],
    ...     params=[im.sym("inp"), im.sym("out")],
    ...     declarations=[],
    ...     body=[
    ...         itir.SetAt(
    ...             expr=im.call("fun1")("inp"),
    ...             domain=im.domain("cartesian_domain", {"IDim": (0, 10)}),
    ...             target=im.ref("out"),
    ...         )
    ...     ],
    ... )
    >>> print(prune_unreferenced_fundefs(program))
    testee(inp, out) {
      fun1 = λ(a) → ·a;
      out @ c⟨ IDimₕ: [0, 10[ ⟩ ← fun1(inp);
    }
    """
    fun_names = [fun.id for fun in program.function_definitions]
    referenced_fun_names = symbol_ref_utils.collect_symbol_refs(program.body, fun_names)

    new_fun_defs = []
    for fun_def in program.function_definitions:
        if fun_def.id in referenced_fun_names:
            new_fun_defs.append(fun_def)

    return itir.Program(
        id=program.id,
        function_definitions=new_fun_defs,
        params=program.params,
        declarations=program.declarations,
        body=program.body,
    )
