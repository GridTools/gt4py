from eve import NodeTranslator
from functional.iterator import ir


BINARY_OPS = {
    "plus": "+",
    "minus": "-",
    "multiplies": "*",
    "divides": "/",
    "eq": "==",
    "less": "<",
    "greater": ">",
    "and_": "and",
    "or_": "or",
}

UNARY_OPS = {
    "deref": "*",
    "lift": "↑",
    "not_": "not ",
}

BINARY_PRECEDENCE = {
    "or": 2,
    "and": 3,
    "==": 4,
    "<": 4,
    ">": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
}

UNARY_PRECEDENCE = {
    "not ": 7,
    "*": 7,
    "↑": 7,
}

OTHER_PRECEDENCE = {
    "lambda": 0,
    "if": 1,
    "call": 8,
}


def _hmerge(*blocks: list[str]):
    def impl(a, b):
        spaces = len(a[-1]) * " "
        return a[:-1] + [a[-1] + b[0]] + [spaces + line for line in b[1:]]

    res = blocks[0]
    for b in blocks[1:]:
        res = impl(res, b)
    return res


def _vmerge(*blocks: list[str]):
    return sum(blocks, [])


def _indent(block: list[str]):
    return ["  " + line for line in block]


SOFT_MAX_LINE_LENGTH = 100


def _cost(block: list[str]):
    max_line_length = max(len(line) for line in block)
    return (
        len(block)
        + max(max_line_length - SOFT_MAX_LINE_LENGTH, 0) * 100
        + sum(max_line_length - len(line) for line in block)
    )


def _optimum(a: list[str], b: list[str]):
    return a if _cost(a) < _cost(b) else b


def _prec_parens(block: list[str], prec: int, op_prec: int):
    if prec > op_prec:
        return _hmerge(["("], block, [")"])
    return block


def _hinterleave(blocks: list[list[str]], sep: str, *, indent: bool = False):
    do_indent = _indent if indent else lambda x: x
    for block in blocks[:-1]:
        yield do_indent(_hmerge(block, [sep]))
    yield do_indent(blocks[-1])


class PrettyPrinter(NodeTranslator):
    def visit_Sym(self, node, *, prec):
        return [node.id]

    def visit_BoolLiteral(self, node, *, prec):
        return [str(node.value)]

    def visit_IntLiteral(self, node, *, prec):
        return [str(node.value)]

    def visit_FloatLiteral(self, node, *, prec):
        return [str(node.value)]

    def visit_OffsetLiteral(self, node, *, prec):
        return [str(node.value)]

    def visit_AxisLiteral(self, node, *, prec):
        return [str(node.value)]

    def visit_SymRef(self, node, *, prec):
        return [node.id]

    def visit_Lambda(self, node, *, prec):
        params = self.visit(node.params, prec=0)
        expr = self.visit(node.expr, prec=0)

        start, bridge = ["λ("], [") → "]
        if not params:
            params = _hmerge(start, bridge)
        else:
            hparams = _hmerge(start, *_hinterleave(params, ", "), bridge)
            vparams = _vmerge(start, *_hinterleave(params, ",", indent=True), bridge)
            params = _optimum(hparams, vparams)

        hbody = _hmerge(params, expr)
        vbody = _vmerge(params, _indent(expr))
        return _prec_parens(_optimum(hbody, vbody), prec, OTHER_PRECEDENCE["lambda"])

    def visit_FunCall(self, node, *, prec):
        if isinstance(node.fun, ir.SymRef):
            if node.fun.id in BINARY_OPS and len(node.args) == 2:
                op = BINARY_OPS[node.fun.id]
                lhs, rhs = self.visit(node.args, prec=BINARY_PRECEDENCE[op])
                h = _hmerge(lhs, [" " + op + " "], rhs)
                v = _vmerge(lhs, _hmerge([op + " "], rhs))
                return _prec_parens(_optimum(h, v), prec, BINARY_PRECEDENCE[op])
            if node.fun.id in UNARY_OPS and len(node.args) == 1:
                op = UNARY_OPS[node.fun.id]
                if (
                    op == "*"
                    and isinstance(node.args[0], ir.FunCall)
                    and isinstance(node.args[0].fun, ir.FunCall)
                    and node.args[0].fun.fun == ir.SymRef(id="shift")
                    and node.args[0].fun.args
                    and isinstance(node.args[0].args[0], ir.SymRef)
                ):
                    # short deref(shift(offsets...)(sym)) → sym[offsets...]
                    assert len(node.args[0].args) == 1
                    expr = self.visit(node.args[0].args[0], prec=OTHER_PRECEDENCE["call"])
                    shifts = self.visit(node.args[0].fun.args, prec=0)
                    res = _hmerge(expr, ["["], *_hinterleave(shifts, ", "), ["]"])
                    return _prec_parens(res, prec, OTHER_PRECEDENCE["call"])
                res = _hmerge([op], self.visit(node.args[0], prec=UNARY_PRECEDENCE[op]))
                return _prec_parens(res, prec, UNARY_PRECEDENCE[op])
            if node.fun.id == "tuple_get" and len(node.args) == 2:
                idx, tup = self.visit(node.args, prec=OTHER_PRECEDENCE["call"])
                res = _hmerge(tup, ["["], idx, ["]"])
                return _prec_parens(res, prec, OTHER_PRECEDENCE["call"])
            if node.fun.id == "named_range" and len(node.args) == 3:
                dim, start, end = self.visit(node.args, prec=0)
                res = _hmerge(dim, [": ["], start, [", "], end, [")"])
                return _prec_parens(res, prec, OTHER_PRECEDENCE["call"])
            if node.fun.id == "domain" and len(node.args) >= 1:
                args = self.visit(node.args, prec=OTHER_PRECEDENCE["call"])
                return _hmerge(["{"], *_hinterleave(args, " × "), ["}"])
            if node.fun.id == "if_" and len(node.args) == 3:
                ifb, thenb, elseb = self.visit(node.args, prec=OTHER_PRECEDENCE["if"])
                hblocks = _hmerge(["if "], ifb, [" then "], thenb, [" else "], elseb)
                vblocks = _vmerge(
                    _hmerge(["if   "], ifb), _hmerge(["then "], thenb), _hmerge(["else "], elseb)
                )
                return _prec_parens(_optimum(hblocks, vblocks), prec, OTHER_PRECEDENCE["if"])

        fun = self.visit(node.fun, prec=OTHER_PRECEDENCE["call"])
        args = self.visit(node.args, prec=0)

        if not args:
            args = [""]
        else:
            hargs = _hmerge(*_hinterleave(args, ", "))
            vargs = _vmerge(*_hinterleave(args, ","))
            args = _optimum(hargs, vargs)

        if node.fun == ir.SymRef(id="make_tuple"):
            htup = _hmerge(["{"], args, ["}"])
            vtup = _vmerge(["{"], _indent(args), ["}"])
            return _optimum(htup, vtup)

        hfun = _hmerge(fun, ["("], args, [")"])
        vfun = _vmerge(_hmerge(fun, ["("]), _indent(args), [")"])
        return _prec_parens(_optimum(hfun, vfun), prec, OTHER_PRECEDENCE["call"])

    def visit_FunctionDefinition(self, node, prec):
        assert prec == 0
        params = self.visit(node.params, prec=0)
        expr = self.visit(node.expr, prec=0)

        start, bridge = [node.id + " = λ("], [") → "]
        if not params:
            params = _hmerge(start, bridge)
        else:
            hparams = _hmerge(start, *_hinterleave(params, ", "), bridge)
            vparams = _vmerge(start, *_hinterleave(params, ",", indent=True), bridge)
            params = _optimum(hparams, vparams)

        hbody = _hmerge(params, expr)
        vbody = _vmerge(params, _indent(expr))
        return _optimum(hbody, vbody)

    def visit_StencilClosure(self, node, *, prec):
        assert prec == 0
        domain = self.visit(node.domain, prec=0)
        stencil = self.visit(node.stencil, prec=0)
        output = self.visit(node.output, prec=0)
        inputs = self.visit(node.inputs, prec=0)

        hinputs = _hmerge(["("], *_hinterleave(inputs, ", "), [")"])
        vinputs = _vmerge(["("], *_hinterleave(inputs, ",", indent=True), [")"])
        inputs = _optimum(hinputs, vinputs)

        head = _hmerge(output, [" ← "])
        foot = _hmerge(inputs, [" @ "], domain)

        h = _hmerge(head, ["("], stencil, [")"], foot)
        v = _vmerge(_hmerge(head, ["("]), _indent(_indent(stencil)), _indent(_hmerge([")"], foot)))
        return _optimum(h, v)

    def visit_FencilDefinition(self, node, *, prec):
        assert prec == 0
        function_definitions = self.visit(node.function_definitions, prec=0)
        closures = self.visit(node.closures, prec=0)
        params = self.visit(node.params, prec=0)

        hparams = _hmerge([node.id + "("], *_hinterleave(params, ", "), [") {"])
        vparams = _vmerge([node.id + "("], *_hinterleave(params, ",", indent=True), [") {"])
        params = _optimum(hparams, vparams)

        function_definitions = _vmerge(*function_definitions)
        closures = _vmerge(*closures)

        return _vmerge(params, _indent(function_definitions), _indent(closures), ["}"])

    @classmethod
    def apply(cls, node):
        return "\n".join(cls().visit(node, prec=0))


pretty_str = PrettyPrinter.apply


def pretty_print(x):
    print(pretty_str(x))
