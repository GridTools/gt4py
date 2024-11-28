# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union

from lark import lark, lexer as lark_lexer, tree as lark_tree, visitors as lark_visitors

from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_specifications as ts


GRAMMAR = """
    start: fencil_definition
        | function_definition
        | declaration
        | stencil_closure
        | set_at
        | if_stmt
        | program
        | prec0

    SYM: CNAME
    SYM_REF: CNAME
    TYPE_LITERAL: CNAME
    INT_LITERAL: SIGNED_INT
    FLOAT_LITERAL: SIGNED_FLOAT
    OFFSET_LITERAL: ( INT_LITERAL | CNAME ) "ₒ"
    AXIS_LITERAL: CNAME ("ᵥ" | "ₕ")
    _literal: INT_LITERAL | FLOAT_LITERAL | OFFSET_LITERAL | AXIS_LITERAL
    ID_NAME: CNAME

    ?prec0: prec1
        | "λ(" ( SYM "," )* SYM? ")" "→" prec0 -> lam

    ?prec1: prec2
        | "if" prec1 "then" prec1 "else" prec1 -> ifthenelse

    ?prec2: prec3
        | prec2 "∨" prec3 -> bool_or  

    ?prec3: prec4
        | prec3 "∧" prec4 -> bool_and

    ?prec4: prec5
        | prec4 "==" prec5 -> eq
        | prec4 "<" prec5 -> less
        | prec4 ">" prec5 -> greater

    ?prec5: prec6
        | prec5 "+" prec6 -> plus
        | prec5 "-" prec6 -> minus

    ?prec6: prec7
        | prec6 "×" prec7 -> multiplies
        | prec6 "/" prec7 -> divides

    ?prec7: prec8
        | "·" prec7 -> deref
        | "¬" prec7 -> bool_not
        | "↑" prec7 -> lift
        | "⇑" prec7 -> as_fieldop

    ?prec8: prec9
        | prec8 "[" prec0 "]" -> tuple_get
        | prec8 "(" ( prec0 "," )* prec0? ")" -> call
        | "{" ( prec0 "," )* prec0? "}" -> make_tuple
        | "⟪" ( prec0 "," )* prec0? "⟫" -> shift
        | "u⟨" ( prec0 "," )* prec0? "⟩" -> unstructured_domain
        | "c⟨" ( prec0 "," )* prec0? "⟩" -> cartesian_domain

    ?prec9: _literal
        | SYM_REF
        | named_range
        | "(" prec0 ")"

    ?stmt: set_at | if_stmt
    set_at: prec0 "@" prec0 "←" prec1 ";"
    else_branch_seperator: "else"
    if_stmt: "if" "(" prec0 ")" "{" ( stmt )* "}" else_branch_seperator "{" ( stmt )* "}"

    named_range: AXIS_LITERAL ":" "[" prec0 "," prec0 ")"
    function_definition: ID_NAME "=" "λ(" ( SYM "," )* SYM? ")" "→" prec0 ";"
    declaration: ID_NAME "=" "temporary(" "domain=" prec0 "," "dtype=" TYPE_LITERAL ")" ";"
    stencil_closure: prec0 "←" "(" prec0 ")" "(" ( SYM_REF ", " )* SYM_REF ")" "@" prec0 ";"
    fencil_definition: ID_NAME "(" ( SYM "," )* SYM ")" "{" ( function_definition )* ( stencil_closure )+ "}"
    program: ID_NAME "(" ( SYM "," )* SYM ")" "{" ( function_definition )* ( declaration )* ( stmt )+ "}"

    %import common (CNAME, SIGNED_FLOAT, SIGNED_INT, WS)
    %ignore WS
"""  # noqa: RUF001 [ambiguous-unicode-character-string]


@lark_visitors.v_args(inline=True)
class ToIrTransformer(lark_visitors.Transformer):
    def SYM(self, value: lark_lexer.Token) -> ir.Sym:
        return ir.Sym(id=value.value)

    def SYM_REF(self, value: lark_lexer.Token) -> Union[ir.SymRef, ir.Literal]:
        if value.value in ("True", "False"):
            return im.literal(value.value, "bool")
        return ir.SymRef(id=value.value)

    def INT_LITERAL(self, value: lark_lexer.Token) -> ir.Literal:
        return im.literal_from_value(int(value.value))

    def FLOAT_LITERAL(self, value: lark_lexer.Token) -> ir.Literal:
        return im.literal(value.value, "float64")

    def TYPE_LITERAL(self, value: lark_lexer.Token) -> ts.TypeSpec:
        if hasattr(ts.ScalarKind, value.upper()):
            return ts.ScalarType(kind=getattr(ts.ScalarKind, value.upper()))
        raise NotImplementedError(f"Type {value} not supported.")

    def OFFSET_LITERAL(self, value: lark_lexer.Token) -> ir.OffsetLiteral:
        v: Union[int, str] = value.value[:-1]
        try:
            v = int(v)
        except ValueError:
            pass
        return ir.OffsetLiteral(value=v)

    def ID_NAME(self, value: lark_lexer.Token) -> str:
        return value.value

    def AXIS_LITERAL(self, value: lark_lexer.Token) -> ir.AxisLiteral:
        name = value.value[:-1]
        kind = ir.DimensionKind.HORIZONTAL if value.value[-1] == "ₕ" else ir.DimensionKind.VERTICAL
        return ir.AxisLiteral(value=name, kind=kind)

    def lam(self, *args: ir.Node) -> ir.Lambda:
        *params, expr = args
        return ir.Lambda(params=params, expr=expr)

    def bool_and(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="and_"), args=[lhs, rhs])

    def bool_or(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="or_"), args=[lhs, rhs])

    def bool_not(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="not_"), args=[arg])

    def bool_xor(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="xor_"), args=[arg])

    def plus(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="plus"), args=[lhs, rhs])

    def minus(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="minus"), args=[lhs, rhs])

    def multiplies(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="multiplies"), args=[lhs, rhs])

    def divides(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="divides"), args=[lhs, rhs])

    def floordiv(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="floordiv"), args=[lhs, rhs])

    def mod(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="mod"), args=[lhs, rhs])

    def eq(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="eq"), args=[lhs, rhs])

    def greater(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="greater"), args=[lhs, rhs])

    def less(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="less"), args=[lhs, rhs])

    def deref(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="deref"), args=[arg])

    def lift(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="lift"), args=[arg])

    def as_fieldop(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="as_fieldop"), args=[arg])

    def astype(self, arg: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="cast_"), args=[arg])

    def shift(self, *offsets: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="shift"), args=list(offsets))

    def tuple_get(self, tup: ir.Expr, idx: ir.Literal) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="tuple_get"), args=[idx, tup])

    def make_tuple(self, *args: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="make_tuple"), args=list(args))

    def named_range(self, name: ir.AxisLiteral, start: ir.Expr, end: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="named_range"), args=[name, start, end])

    def cartesian_domain(self, *ranges: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="cartesian_domain"), args=list(ranges))

    def unstructured_domain(self, *ranges: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="unstructured_domain"), args=list(ranges))

    def ifthenelse(self, condition: ir.Expr, then: ir.Expr, otherwise: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=ir.SymRef(id="if_"), args=[condition, then, otherwise])

    def call(self, fun: ir.Expr, *args: ir.Expr) -> ir.FunCall:
        return ir.FunCall(fun=fun, args=list(args))

    def function_definition(self, *args: ir.Node) -> ir.FunctionDefinition:
        fid, *params, expr = args
        return ir.FunctionDefinition(id=fid, params=params, expr=expr)

    def stencil_closure(self, *args: ir.Expr) -> ir.StencilClosure:
        output, stencil, *inputs, domain = args
        return ir.StencilClosure(domain=domain, stencil=stencil, output=output, inputs=inputs)

    def if_stmt(self, cond: ir.Expr, *args):
        found_else_seperator = False
        true_branch = []
        false_branch = []
        for arg in args:
            if isinstance(arg, lark_tree.Tree):
                assert arg.data == "else_branch_seperator"
                found_else_seperator = True
                continue

            if not found_else_seperator:
                true_branch.append(arg)
            else:
                false_branch.append(arg)

        return ir.IfStmt(
            cond=cond,
            true_branch=true_branch,
            false_branch=false_branch,
        )

    def declaration(self, *args: ir.Expr) -> ir.Temporary:
        tid, domain, dtype = args
        return ir.Temporary(id=tid, domain=domain, dtype=dtype)

    def set_at(self, *args: ir.Expr) -> ir.SetAt:
        target, domain, expr = args
        return ir.SetAt(expr=expr, domain=domain, target=target)

    # TODO(havogt): remove after refactoring.
    def fencil_definition(self, fid: str, *args: ir.Node) -> ir.FencilDefinition:
        params = []
        function_definitions = []
        closures = []
        for arg in args:
            if isinstance(arg, ir.Sym):
                params.append(arg)
            elif isinstance(arg, ir.FunctionDefinition):
                function_definitions.append(arg)
            else:
                assert isinstance(arg, ir.StencilClosure)
                closures.append(arg)
        return ir.FencilDefinition(
            id=fid, function_definitions=function_definitions, params=params, closures=closures
        )

    def program(self, fid: str, *args: ir.Node) -> ir.Program:
        params = []
        function_definitions = []
        body = []
        declarations = []
        for arg in args:
            if isinstance(arg, ir.Sym):
                params.append(arg)
            elif isinstance(arg, ir.FunctionDefinition):
                function_definitions.append(arg)
            elif isinstance(arg, ir.Temporary):
                declarations.append(arg)
            else:
                assert isinstance(arg, ir.Stmt)
                body.append(arg)
        return ir.Program(
            id=fid,
            function_definitions=function_definitions,
            params=params,
            body=body,
            declarations=declarations,
        )

    def start(self, arg: ir.Node) -> ir.Node:
        return arg


def pparse(pretty_str: str) -> ir.Node:
    parser = lark.Lark(GRAMMAR, parser="earley")
    tree = parser.parse(pretty_str)
    return ToIrTransformer(visit_tokens=True).transform(tree)
