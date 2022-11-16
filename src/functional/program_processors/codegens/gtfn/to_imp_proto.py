import dataclasses
from functional.program_processors.codegens.gtfn import gtfn_ir
from eve import NodeVisitor, NodeTranslator
from eve.utils import UIDGenerator
from functional.program_processors.codegens.gtfn.itir_to_gtfn_ir import pytype_to_cpptype

from typing import List


@dataclasses.dataclass(frozen=True)
class ToImp(NodeVisitor):
    imp_list: List[str]
    # idx: int
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    _builtins_mapping = {
        "abs": "std::abs",
        "sin": "std::sin",
        "cos": "std::cos",
        "tan": "std::tan",
        "arcsin": "std::asin",
        "arccos": "std::acos",
        "arctan": "std::atan",
        "sinh": "std::sinh",
        "cosh": "std::cosh",
        "tanh": "std::tanh",
        "arcsinh": "std::asinh",
        "arccosh": "std::acosh",
        "arctanh": "std::atanh",
        "sqrt": "std::sqrt",
        "exp": "std::exp",
        "log": "std::log",
        "gamma": "std::tgamma",
        "cbrt": "std::cbrt",
        "isfinite": "std::isfinite",
        "isinf": "std::isinf",
        "isnan": "std::isnan",
        "floor": "std::floor",
        "ceil": "std::ceil",
        "trunc": "std::trunc",
        "minimum": "std::min",
        "maximum": "std::max",
        "fmod": "std::fmod",
        "power": "std::pow",
    }

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

    def visit_OffsetLiteral(self, node: gtfn_ir.OffsetLiteral) -> str:
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

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

    @staticmethod
    def _depth(node: gtfn_ir.FunCall) -> int:
        # TODO bad hardcoded tring
        return (
            1 + ToImp._depth(node.args[0])
            if isinstance(node.args[0], gtfn_ir.FunCall) and "step" in node.args[0].fun.id
            else 0
        )

    def visit_Lambda(self, node: gtfn_ir.Lambda, **kwargs) -> str:
        idx_to_replace = node.params[1].id  # find _i_X parameter

        class Replace(NodeTranslator):
            def visit_SymRef(self, node):
                if node.id == idx_to_replace:
                    return gtfn_ir.OffsetLiteral(value=self.cur_idx)
                return self.generic_visit(node)

            def __init__(self, cur_idx: int):
                self.cur_idx = cur_idx

        for lambda_iter in range(kwargs["num_iter"]):
            new_expr = Replace(cur_idx=lambda_iter).visit(node.expr.rhs)
            rhs = self.visit(new_expr)
            self.imp_list.append(f"{kwargs['red_idx']} += {rhs};")

    def visit_FunCall(self, node: gtfn_ir.FunCall) -> str:
        if (
            isinstance(node.fun, gtfn_ir.Lambda) and "step" in node.fun.params[0].id
        ):  # TODO: bad hardcoded string
            red_idx = self.uids.sequential_id(prefix="red")
            # this can be improved by looking at the type of the initial condition
            # (first argument to innermost lambda)
            # for now, let's just guess double
            self.imp_list.append(f"double {red_idx} = 0.;")
            num_iter = 1 + ToImp._depth(node.fun.expr)
            self.visit(node.args[0], num_iter=num_iter, red_idx=red_idx)
            return red_idx
        if isinstance(node.fun, gtfn_ir.Lambda):
            lam_idx = self.uids.sequential_id(prefix="lam")
            params = [self.visit(param) for param in node.fun.params]
            args = [self.visit(arg) for arg in node.args]
            for param, arg in zip(params, args):
                self.imp_list.append(f"auto {param} = {arg};")
            expr = self.visit(node.fun.expr)
            self.imp_list.append(f"auto {lam_idx} = {expr};")
            return lam_idx
        if (
            isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id == "make_tuple"
        ):  # TODO: bad hardcoded string
            tupl_idx = self.uids.sequential_id(prefix="tupl")
            for i, arg in enumerate(node.args):
                expr = self.visit(arg)
                self.imp_list.append(f"auto {tupl_idx}_{i} = {expr};")
            tup_args = ",".join([f"{tupl_idx}_{i}" for i in range(len(node.args))])
            self.imp_list.append(f"auto {tupl_idx} = make_tuple({tup_args});")
            return tupl_idx
        if isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id in gtfn_ir.GTFN_BUILTINS:
            qualified_fun_name = f"{node.fun.id}"
            args = [self.visit(arg) for arg in node.args]
            return f"{qualified_fun_name}({', '.join(args)})"
        if isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id in self._builtins_mapping:
            fun_name = self._builtins_mapping[node.fun.id]
            args = [self.visit(arg) for arg in node.args]
            return f"{fun_name}({', '.join(args)})"
        raise NotImplementedError(f"unhandled function call {node}")

    def visit_TernaryExpr(self, node: gtfn_ir.TernaryExpr) -> str:
        cond = self.visit(node.cond)
        if_ = self.visit(node.true_expr)
        else_ = self.visit(node.false_expr)
        cond_idx = self.uids.sequential_id(prefix="cond")
        # this just guesses double as type of temporary
        self.imp_list.append(
            f"""
      double {cond_idx} = 0;
      if ({cond}) {{
        {cond_idx} = {if_};                
      }} else {{
        {cond_idx} = {else_};
      }}
      """
        )
        return cond_idx

    # def __init__(self):
    #     self.imp_list = []


def to_imp(node: gtfn_ir.FunctionDefinition):
    to_imp = ToImp(imp_list=[])
    ret = to_imp.visit(node.expr)
    to_imp.imp_list.append(f"return {ret}")
    return "\n".join(to_imp.imp_list)
