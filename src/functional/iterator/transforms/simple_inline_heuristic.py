from collections.abc import Callable

from eve import NodeVisitor
from functional.iterator import ir


class AnalyzeLifts(NodeVisitor):
    def visit_FunCall(
        self, node: ir.FunCall, *, inline: dict[int, bool], force_noinline: bool = False
    ) -> None:
        if node.fun == ir.SymRef(id="lift"):
            (arg,) = node.args
            if isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="scan"):
                inline[id(node)] = False
                return self.generic_visit(node, inline=inline, force_noinline=True)
            else:
                inline[id(node)] = not force_noinline
                return self.generic_visit(node, inline=inline, force_noinline=False)
        elif (
            isinstance(node.fun, ir.FunCall)
            and node.fun.fun == ir.SymRef(id="lift")
            and isinstance(node.fun.args[0], ir.FunCall)
            and node.fun.args[0].fun == ir.SymRef(id="scan")
        ):
            return self.generic_visit(node, inline=inline, force_noinline=True)
        self.generic_visit(node, inline=inline, force_noinline=force_noinline)


def heuristic(node: ir.Node) -> Callable[[ir.Node], bool]:
    """Create a predicate for the InlineLifts transformation.

    Follows the simple rules:
    - Don’t inline scans (as there is no efficient way to inline them, also required by some
      backends, e.g. gtfn)
    - Don’t inline the first lifted function call within a scan (otherwise, all stencils would get
      inlined into the scans, leading to reduced parallelism/scan-only computation)
    """
    inline = dict[int, bool]()
    AnalyzeLifts().visit(node, inline=inline)

    def predicate(node: ir.Node) -> bool:
        return inline[id(node)]

    return predicate
