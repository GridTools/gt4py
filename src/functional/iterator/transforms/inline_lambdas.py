import dataclasses
from typing import Optional

from eve import NodeTranslator
from functional.iterator import ir
from functional.iterator.transforms.count_symbol_refs import CountSymbolRefs
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs, RenameSymbols


def inline_lambda(
    node: ir.FunCall,
    opcount_preserving=False,
    force_inline_lift=False,
    eligible_params: Optional[list[bool]] = None,
):
    assert isinstance(node.fun, ir.Lambda)
    eligible_params = eligible_params or [True] * len(node.fun.params)

    assert len(eligible_params) == len(node.fun.params) == len(node.args)

    if opcount_preserving:
        ref_counts = CountSymbolRefs.apply(node.fun.expr, [p.id for p in node.fun.params])

        for i, param in enumerate(node.fun.params):
            # TODO(tehrengruber): allow inlining more complicated zero-op expressions like
            #  ignore_shift(...)(it_sym)  # noqa: E800
            if ref_counts[param.id] != 1 and not isinstance(
                node.args[i], (ir.SymRef, ir.Literal, ir.OffsetLiteral)
            ):
                eligible_params[i] = False

    if force_inline_lift:
        for i, arg in enumerate(node.args):
            if (
                isinstance(arg, ir.FunCall)
                and isinstance(arg.fun, ir.FunCall)
                and isinstance(arg.fun.fun, ir.SymRef)
                and arg.fun.fun.id in ["lift", "translate_shift", "ignore_shift"]
            ):
                eligible_params[i] = True

    if node.fun.params and not any(eligible_params):
        return node

    refs = set().union(
        *(
            arg.pre_walk_values().if_isinstance(ir.SymRef).getattr("id").to_set()
            for arg, eligible in zip(node.args, eligible_params)
            if eligible
        )
    )
    syms = node.fun.expr.pre_walk_values().if_isinstance(ir.Sym).getattr("id").to_set()
    clashes = refs & syms
    expr = node.fun.expr
    if clashes:
        # TODO(tehrengruber): find a better way of generating new symbols
        #  in `name_map` that don't collide with each other. E.g. this
        #  must still work:
        # (lambda arg, arg_: (lambda arg_: ...)(arg))(a, b)  # noqa: E800
        name_map = {}

        def new_name(name):
            while name in refs or name in syms or name in name_map.values():
                name += "_"
            return name

        for sym in clashes:
            name_map[sym] = new_name(sym)

        expr = RenameSymbols().visit(expr, name_map=name_map)

    symbol_map = {
        param.id: arg
        for param, arg, eligible in zip(node.fun.params, node.args, eligible_params)
        if eligible
    }
    new_expr = RemapSymbolRefs().visit(expr, symbol_map=symbol_map)

    if all(eligible_params):
        return new_expr
    else:
        return ir.FunCall(
            fun=ir.Lambda(
                params=[
                    param
                    for param, eligible in zip(node.fun.params, eligible_params)
                    if not eligible
                ],
                expr=new_expr,
            ),
            args=[arg for arg, eligible in zip(node.args, eligible_params) if not eligible],
        )


@dataclasses.dataclass
class InlineLambdas(NodeTranslator):
    """Inline lambda calls by substituting every argument by its value."""

    opcount_preserving: bool

    force_inline_lift: bool

    @classmethod
    def apply(cls, node: ir.Node, opcount_preserving=False, force_inline_lift=False):
        """
        Inline lambda calls by substituting every arguments by its value.

        Examples:
            `(λ(x) → x)(y)` to `y`
            `(λ(x) → x)(y+y)` to `y+y`
            `(λ(x) → x+x)(y+y)` to `y+y+y+y` if not opcount_preserving
            `(λ(x) → x+x)(y+y)` stays as is if opcount_preserving

        Arguments:
            opcount_preserving: Preserve the number of operations, i.e. only
            inline lambda call if the resulting call has the same number of
            operations.
        """
        return cls(
            opcount_preserving=opcount_preserving,
            force_inline_lift=force_inline_lift,
        ).visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            return inline_lambda(
                node,
                opcount_preserving=self.opcount_preserving,
                force_inline_lift=self.force_inline_lift,
            )

        return node
