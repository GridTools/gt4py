import textwrap
from typing import Any

import lark

from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import TemplatedGenerator
from functional.iterator import ir


class ToLisp(TemplatedGenerator):
    Sym = as_fmt("{id}")
    BoolLiteral = as_fmt("{'#t' if value == 'True' else '#f'}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    StringLiteral = as_fmt('"{value}"')
    NoneLiteral = as_fmt("gt-none")
    OffsetLiteral = as_fmt(
        "(gt-offset {'\"' + value + '\"' if isinstance(_this_node.value, str) else value})"
    )
    AxisLiteral = as_fmt('(gt-axis "{value}")')
    SymRef = as_fmt("{id}")
    Lambda = as_fmt("(gt-lambda ({' '.join(params)}) {expr})")
    FunCall = as_fmt("({fun} {' '.join(args)})")
    FunctionDefinition = as_fmt("(gt-function {id} ({' '.join(params)}) {expr})")
    Program = as_fmt("(gt-program ({''.join(function_definitions)}) {''.join(fencil_definitions)})")
    StencilClosure = as_fmt("(gt-stencil-closure {domain} {stencil} {output} {' '.join(inputs)})")
    FencilDefinition = as_fmt("(gt-fencil {id} ({' '.join(params)}) {''.join(closures)})")

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
    _atom: BOOL | ID | ESCAPED_STRING | FLOAT | INTEGER | SYM

    SPECIAL: "!" | "$" | "_" | "-" | "." | "/" | ":" | "?" | "+" | "<" | "=" | ">" | "%" | "&" | "*" | "@" | "`" | "^" | "~"
    IDCHAR: DIGIT | LETTER | SPECIAL
    INTEGER.2: SIGNED_INT
    FLOAT.3: SIGNED_FLOAT
    BOOL: "#" ("t" | "f")
    ID: IDCHAR+
    SYM: "'" IDCHAR+

    %import common (DIGIT, ESCAPED_STRING, LETTER, SIGNED_INT, SIGNED_FLOAT, WS)
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
            if elements[0].id == "gt-offset":
                return ir.OffsetLiteral(
                    value=elements[1].id if hasattr(elements[1], "id") else elements[1].value
                )
            if elements[0].id == "gt-axis":
                return ir.AxisLiteral(value=elements[1].value)
            if elements[0].id == "gt-lambda":
                return ir.Lambda(
                    params=[ir.Sym(id=p.id) for p in elements[1]], expr=to_funcall(elements[2])
                )
            if elements[0].id == "gt-function":
                return ir.FunctionDefinition(
                    id=elements[1].id,
                    params=[ir.Sym(id=p.id) for p in elements[2]],
                    expr=to_funcall(elements[3]),
                )
            if elements[0].id == "gt-program":
                return ir.Program(
                    function_definitions=elements[1],
                    fencil_definitions=elements[2:],
                )
            if elements[0].id == "gt-stencil-closure":
                return ir.StencilClosure(
                    domain=to_funcall(elements[1]),
                    stencil=elements[2],
                    output=elements[3],
                    inputs=list(elements[4:]),
                )
            if elements[0].id == "gt-fencil":
                return ir.FencilDefinition(
                    id=elements[1].id,
                    params=[ir.Sym(id=p.id) for p in elements[2]],
                    closures=list(elements[3:]),
                )
        return elements

    def BOOL(self, value):
        return ir.BoolLiteral(value=value.value == "#t")

    def ID(self, value):
        if value.value == "gt-none":
            return ir.NoneLiteral()
        return ir.SymRef(id=value.value)

    def ESCAPED_STRING(self, value):
        return ir.StringLiteral(value=value.value[1:-1])

    def INTEGER(self, value):
        return ir.IntLiteral(value=int(value.value))

    def FLOAT(self, value):
        return ir.FloatLiteral(value=float(value.value))

    def SYM(self, value):
        return ir.SymRef(id=value.value[1:])


def lisp_to_ir(lisp_str):
    parser = lark.Lark(GRAMMAR, parser="lalr", transformer=ToIrTransformer())
    return parser.parse(lisp_str)


def lisp_to_ir_using_lisp(lisp_str):
    import pathlib
    import subprocess

    scm_script = pathlib.Path(__file__).parent.absolute() / "lisp_to_ir.scm"
    python_code = subprocess.run(
        ["scheme", "--quiet", "--load", scm_script],
        input=lisp_str,
        capture_output=True,
        check=True,
        encoding="ascii",
    ).stdout
    return eval(python_code)


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
