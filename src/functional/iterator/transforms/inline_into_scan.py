import eve
from eve import NodeTranslator, traits
from functional.iterator import ir


def _is_scan(node: ir.Node):
    return (
        isinstance(node, ir.FunCall)
        and isinstance(node.fun, ir.FunCall)
        and node.fun.fun == ir.SymRef(id="scan")
    )


def _is_userdefined_symbolref(node: ir.Expr, symtable: dict[eve.SymbolName, ir.Sym]):
    return (
        isinstance(node, ir.SymRef)
        and node.id in symtable
        and node.id
        not in [
            n.id for n in ir.FencilDefinition._NODE_SYMBOLS_
        ]  # TODO this is relatively expensive, should we provide a way to exclude non-userdefined builtins from eve?
    )


def _extract_symrefs(
    nodes: list[ir.Expr], symtable: dict[eve.SymbolName, ir.Sym]
) -> set[ir.SymRef]:
    symrefs = set()
    for n in nodes:
        if isinstance(n, ir.SymRef):
            if _is_userdefined_symbolref(n, symtable):
                symrefs.add(n)
        else:
            symrefs.update(
                n.pre_walk_values()  # type: ignore [arg-type]
                .if_isinstance(ir.SymRef)
                .filter(lambda x: _is_userdefined_symbolref(x, symtable))
                .to_set()
            )
    return symrefs


class InlineIntoScan(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Inline non-SymRef arguments into the scan.

    example:
    scan(λ(state, isym0, isym1) → body(state, isym0, isym1), forward, init)(sym0, f(sym0,sym1,sym2))
    to
    scan(λ(state, sym0, sym1, sym2) → (λ(isym0, isym1) → body(state, isym0, isym1))(sym0, f(sym0,sym1,sym2)), forward, init)(sym0, sym1,sym2)

    algorithm:
    - take args of scan: `sym0`, `f(sym0, sym1, sym2)`
    - extract all symrefs that are not builtins: `sym0`, `sym1`, `sym2`
    - create a lambda with first (state/carry) param taken from original scanpass (`state`) and new Symbols with the name of the extracted symrefs: `λ(state, sym0, sym1, sym2)`
    - the body is a call to the original scanpass, but with `state` param removed `λ(isym0, isym1) → body(state, isym0, isym1)` (`state` is captured)
    - it is called with the original args of the scan: `sym0, f(sym0,sym1,sym2)`
    - wrap the new scanpass in a scan call with the original `forward` and `init`
    - call it with the extrated symrefs
    - note: there is no symbol clash, re-used
    """

    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if _is_scan(node):
            original_scan_args = node.args
            original_scan_call = node.fun
            assert isinstance(original_scan_call, ir.FunCall)
            refs_in_args = _extract_symrefs(original_scan_args, kwargs["symtable"])
            original_scanpass = original_scan_call.args[0]
            assert isinstance(original_scanpass, (ir.FunctionDefinition, ir.Lambda))

            new_scanpass = ir.Lambda(
                params=[
                    original_scanpass.params[0],
                    *(ir.Sym(id=ref.id) for ref in refs_in_args),
                ],
                expr=ir.FunCall(
                    fun=ir.Lambda(
                        params=[*original_scanpass.params[1:]], expr=original_scanpass.expr
                    ),
                    args=original_scan_args,
                ),
            )
            new_scan = ir.FunCall(
                fun=ir.SymRef(id="scan"), args=[new_scanpass, *original_scan_call.args[1:]]
            )
            result = ir.FunCall(fun=new_scan, args=[*refs_in_args])
            return result
        return self.generic_visit(node, **kwargs)


class TupleMerger(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if node.fun == ir.SymRef(id="make_tuple") and all(
            isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="tuple_get")
            for arg in node.args
        ):
            assert isinstance(node.args[0], ir.FunCall)
            first_expr = node.args[0].args[1]
            for i, v in enumerate(node.args):
                assert isinstance(v, ir.FunCall)
                assert isinstance(v.args[0], ir.Literal)
                if not (int(v.args[0].value) == i and v.args[1] == first_expr):
                    return self.generic_visit(node)
            # TODO at this point we don't know if the inner tuple has maybe more elements than the tuple we are removing

            return first_expr
        return self.generic_visit(node)
