from functional.program_processors.codegens.gtfn import gtfn_ir
from eve import NodeVisitor
from functional.program_processors.codegens.gtfn.itir_to_gtfn_ir import pytype_to_cpptype

from typing import List


class ToImp(NodeVisitor):
    imp_list: List[str]
    idx: int

    @staticmethod
    def asfloat(value: str) -> str:
        if "." not in value and "e" not in value and "E" not in value:
            return f"{value}."
        return value

    def visit_Sym(self, node: gtfn_ir.Sym) -> str:
        return str(node.id)

    def visit_UnaryExpr(self, node: gtfn_ir.UnaryExpr) -> str:
        return f"{node.op} {self.visit(node.expr)}"

    def visit_BinaryExpr(self, node: gtfn_ir.BinaryExpr) -> str:
        return f"{self.visit(node.lhs)} {node.op} {self.visit(node.rhs)}"

    def visit_Literal(self, node: gtfn_ir.Literal) -> str:
        match pytype_to_cpptype(node.type):
            case "int":
                return node.value + "_c"
            case "float":
                return self.asfloat(node.value) + "f"
            case "double":
                return self.asfloat(node.value)
            case "bool":
                return node.value.lower()
            case _:
                return node.value

    def visit_SymRef(self, node: gtfn_ir.SymRef) -> str:
        return node.id

    def visit_FunCall(self, node: gtfn_ir.FunCall) -> str:
        if isinstance(node.fun, gtfn_ir.Lambda):
            params = [self.visit(param) for param in node.fun.params]
            args = [self.visit(arg) for arg in node.args]
            for param, arg in zip(params, args):
                self.imp_list.append(f"auto {param} = {arg};")
            expr = self.visit(node.fun.expr)
            self.imp_list.append(f"auto lam{self.idx} = {expr};")
            self.idx += 1
            return f"lam{self.idx}"
        if isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id == "make_tuple":
            for i, arg in enumerate(node.args):
                expr = self.visit(arg)
                self.imp_list.append(f"auto tupl_{self.idx}_{i} = {expr};")
            tup_args = ",".join([f"tupl_{self.idx}_{i}" for i in range(len(node.args))])
            self.imp_list.append(f"auto tupl_{self.idx} = make_tuple({tup_args});")
            self.idx += 1
            return f"tupl_{self.idx}"
        if isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id in gtfn_ir.GTFN_BUILTINS:
            qualified_fun_name = f"gtfn::{node.fun.id}"
            args = [self.visit(arg) for arg in node.args]
            return f"{qualified_fun_name}({', '.join(args)})"
        print("UNHANDLED FunCall")

    def visit_TernaryExpr(self, node: gtfn_ir.TernaryExpr) -> str:
        cond = self.visit(node.cond)
        if_ = self.visit(node.true_expr)
        else_ = self.visit(node.false_expr)
        idx = self.idx
        self.imp_list.append(
            f"""
      double cond{idx} = 0;
      if ({cond}) {{
        cond{idx} = {if_};                
      }} else {{
        cond{idx} = {else_};
      }}
      """
        )
        self.idx += 1
        return f"cond{idx}"

    def __init___(self):
        self.idx = 0
        self.imp_list = []


def to_imp(node: gtfn_ir.FunctionDefinition):
    to_imp = ToImp()
    to_imp.idx = 0
    to_imp.imp_list = []
    ret = to_imp.visit(node.expr)
    to_imp.imp_list.append(f"return {ret};")
    return "\n".join(to_imp.imp_list)
