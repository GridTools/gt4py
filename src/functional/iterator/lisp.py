import textwrap
from typing import Union, cast

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
    StencilClosure = as_fmt("(gt-stencil-closure {domain} {stencil} {output} {' '.join(inputs)})")
    FencilDefinition = as_fmt(
        "(gt-fencil {id} ({' '.join(function_definitions)}) ({' '.join(params)}) {''.join(closures)})"
    )


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
    def selist(self, *elements: Union[ir.Node, tuple]) -> Union[ir.Node, tuple]:
        def to_funcall(elems: Union[ir.Node, tuple]) -> ir.Node:
            if not isinstance(elems, tuple):
                return elems
            return ir.FunCall(fun=to_funcall(elems[0]), args=[to_funcall(e) for e in elems[1:]])

        if elements and isinstance(elements[0], ir.SymRef):
            if elements[0].id == "gt-offset":
                assert isinstance(elements[1], (ir.IntLiteral, ir.StringLiteral))
                return ir.OffsetLiteral(value=elements[1].value)
            if elements[0].id == "gt-axis":
                assert isinstance(elements[1], ir.StringLiteral)
                return ir.AxisLiteral(value=elements[1].value)
            if elements[0].id == "gt-lambda":
                params = cast(tuple[ir.SymRef], elements[1])
                return ir.Lambda(
                    params=[ir.Sym(id=p.id) for p in params], expr=to_funcall(elements[2])
                )
            if elements[0].id == "gt-function":
                assert isinstance(elements[1], ir.SymRef)
                params = cast(tuple[ir.SymRef], elements[2])
                return ir.FunctionDefinition(
                    id=elements[1].id,
                    params=params,
                    expr=to_funcall(elements[3]),
                )
            if elements[0].id == "gt-stencil-closure":
                return ir.StencilClosure(
                    domain=to_funcall(elements[1]),
                    stencil=elements[2],
                    output=elements[3],
                    inputs=list(elements[4:]),
                )
            if elements[0].id == "gt-fencil":
                assert isinstance(elements[1], ir.SymRef)
                params = cast(tuple[ir.SymRef], elements[3])
                return ir.FencilDefinition(
                    id=elements[1].id,
                    function_definitions=elements[2],
                    params=params,
                    closures=list(elements[4:]),
                )
        return elements

    def BOOL(self, value: lark.Token) -> ir.BoolLiteral:
        return ir.BoolLiteral(value=value.value == "#t")

    def ID(self, value: lark.Token) -> Union[ir.NoneLiteral, ir.SymRef]:
        if value.value == "gt-none":
            return ir.NoneLiteral()
        return ir.SymRef(id=value.value)

    def ESCAPED_STRING(self, value: lark.Token) -> ir.StringLiteral:
        return ir.StringLiteral(value=value.value[1:-1])

    def INTEGER(self, value: lark.Token) -> ir.IntLiteral:
        return ir.IntLiteral(value=int(value.value))

    def FLOAT(self, value: lark.Token) -> ir.FloatLiteral:
        return ir.FloatLiteral(value=float(value.value))

    def SYM(self, value: lark.Token) -> ir.SymRef:
        return ir.SymRef(id=value.value[1:])


def lisp_to_ir(lisp_str: str) -> ir.Node:
    parser = lark.Lark(GRAMMAR, parser="lalr", transformer=ToIrTransformer())
    return cast(ir.Node, parser.parse(lisp_str))


def lisp_to_ir_using_lisp(lisp_str: str) -> ir.Node:
    import pathlib
    import subprocess

    scm_script = pathlib.Path(__file__).parent.absolute() / "lisp_to_ir.scm"
    # Currently expecting GNU/MIT scheme
    python_code = subprocess.run(
        ["scheme", "--quiet", "--load", scm_script],
        input="#!no-fold-case\n" + lisp_str,
        capture_output=True,
        check=True,
        encoding="ascii",
    ).stdout
    return eval(python_code)


@lark.v_args(inline=True)
class PrettyFormatter(lark.Transformer):
    def selist(self, *elements: str) -> str:
        single_line = "(" + " ".join(elements) + ")"
        maxlen = 100
        first_break = single_line.find("\n")
        if first_break > maxlen or first_break == -1 and len(single_line) > maxlen:
            return "(\n" + "\n".join(textwrap.indent(e, "  ") for e in elements) + "\n)"
        return single_line

    def CNAME(self, value: lark.Token) -> str:
        return value.value

    def ESCAPED_STRING(self, value: lark.Token) -> str:
        return value.value

    def SIGNED_INT(self, value: lark.Token) -> str:
        return value.value

    def SIGNED_FLOAT(self, value: lark.Token) -> str:
        return value.value


def pretty_format(lisp_str: str) -> str:
    parser = lark.Lark(GRAMMAR, parser="lalr", transformer=PrettyFormatter())
    return cast(str, parser.parse(lisp_str))
