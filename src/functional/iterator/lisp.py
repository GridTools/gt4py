import textwrap
from typing import Any

import lark

from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import TemplatedGenerator
from functional.iterator import ir


class ToLisp(TemplatedGenerator):
    Sym = as_fmt("{id}")
    BoolLiteral = as_fmt("{str(value).lower()}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    StringLiteral = as_fmt('"{value}"')
    NoneLiteral = as_fmt("none")
    OffsetLiteral = as_fmt("(offset {value})")
    AxisLiteral = as_fmt("(axis {value})")
    SymRef = as_fmt("{id}")
    Lambda = as_fmt("(lambda ({' '.join(params)}) {expr})")
    FunCall = as_fmt("({fun} {' '.join(args)})")
    FunctionDefinition = as_fmt("(defun {id} ({' '.join(params)}) {expr})")
    Program = as_fmt(
        "(program ({''.join(function_definitions)}) ({''.join(fencil_definitions)}) ({''.join(setqs)}))"
    )
    Setq = as_fmt("(setq {id} {expr})")
    StencilClosure = as_fmt("(stencil_closure {domain} {stencil} {output} {' '.join(inputs)})")
    FencilDefinition = as_fmt("(fencil {id} ({' '.join(params)}) {''.join(closures)})")

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        try:
            from yasi import indent_code

            indented = indent_code(generated_code, "--dialect lisp")
            return "".join(indented["indented_code"])
        except ImportError:
            return generated_code


ir_to_lisp = ToLisp.apply


GRAMMAR = r"""
    ?start: _sexpr
    _sexpr: selist | _atom
    selist: "(" _sexpr* ")"
    _atom: CNAME | ESCAPED_STRING | SIGNED_INT | SIGNED_FLOAT

    %import common (CNAME, ESCAPED_STRING, SIGNED_INT, SIGNED_FLOAT, WS)
    %ignore WS
"""


@lark.v_args(inline=True)
class ToIrTransformer(lark.Transformer):
    def selist(self, *elements):
        def to_funcall(elems):
            if not isinstance(elems, tuple):
                return elems
            return ir.FunCall(fun=to_funcall(elems[0]), args=[to_funcall(e) for e in elems[1:]])

        if elements and isinstance(elements[0], ir.SymRef):
            if elements[0].id == "offset":
                return ir.OffsetLiteral(
                    value=elements[1].id if hasattr(elements[1], "id") else elements[1].value
                )
            if elements[0].id == "axis":
                return ir.AxisLiteral(value=elements[1].id)
            if elements[0].id == "lambda":
                return ir.Lambda(
                    params=[ir.Sym(id=p.id) for p in elements[1]], expr=to_funcall(elements[2])
                )
            if elements[0].id == "defun":
                return ir.FunctionDefinition(
                    id=elements[1].id,
                    params=[ir.Sym(id=p.id) for p in elements[2]],
                    expr=to_funcall(elements[3]),
                )
            if elements[0].id == "program":
                return ir.Program(
                    function_definitions=elements[1],
                    fencil_definitions=elements[2],
                    setqs=elements[3],
                )
            if elements[0].id == "setq":
                return ir.Setq(id=elements[1].id, expr=to_funcall(elements[2]))
            if elements[0].id == "stencil_closure":
                return ir.StencilClosure(
                    domain=to_funcall(elements[1]),
                    stencil=elements[2],
                    output=elements[3],
                    inputs=list(elements[4:]),
                )
            if elements[0].id == "fencil":
                return ir.FencilDefinition(
                    id=elements[1].id,
                    params=[ir.Sym(id=p.id) for p in elements[2]],
                    closures=list(elements[3:]),
                )
        return elements

    def CNAME(self, value):
        if value.value == "true":
            return ir.BoolLiteral(value=True)
        if value.value == "false":
            return ir.BoolLiteral(value=False)
        if value.value == "none":
            return ir.NoneLiteral()
        return ir.SymRef(id=value.value)

    def ESCAPED_STRING(self, value):
        return ir.StringLiteral(value=value.value[1:-1])

    def SIGNED_INT(self, value):
        return ir.IntLiteral(value=int(value.value))

    def SIGNED_FLOAT(self, value):
        return ir.FloatLiteral(value=float(value.value))


def lisp_to_ir(lisp_str):
    parser = lark.Lark(GRAMMAR, parser="lalr", transformer=ToIrTransformer())
    return parser.parse(lisp_str)


@lark.v_args(inline=True)
class PrettyFormatter(lark.Transformer):
    def selist(self, *elements):
        single_line = "(" + " ".join(elements) + ")"
        maxlen = 100
        first_break = single_line.find("\n")
        if first_break > maxlen or first_break == -1 and len(single_line) > maxlen:
            return "(\n" + "\n".join(textwrap.indent(e, "  ") for e in elements) + "\n)"
        return single_line

    def CNAME(self, value):
        return value.value

    def ESCAPED_STRING(self, value):
        return value.value

    def SIGNED_INT(self, value):
        return value.value

    def SIGNED_FLOAT(self, value):
        return value.value


def pretty_format(lisp_str):
    parser = lark.Lark(GRAMMAR, parser="lalr", transformer=PrettyFormatter())
    return parser.parse(lisp_str)
