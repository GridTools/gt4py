import dataclasses
from functional.program_processors.codegens.gtfn import gtfn_ir
import eve
from eve import NodeVisitor, NodeTranslator
from eve.utils import UIDGenerator
from functional.program_processors.codegens.gtfn.gtfn_im_ir import (
    Stmt,
    InitStmt,
    AssignStmt,
    ReturnStmt,
    Conditional,
    ImperativeFunctionDefinition,
)

from typing import List, Union, Any


@dataclasses.dataclass(frozen=True)
class ToImpIR(NodeVisitor):
    imp_list_ir: List[Union[Stmt, Conditional]]
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @staticmethod
    def asfloat(value: str) -> str:
        if "." not in value and "e" not in value and "E" not in value:
            return f"{value}."
        return value

    def visit_Node(self, node):
        return node

    def visit_UnaryExpr(self, node: gtfn_ir.UnaryExpr):
        return gtfn_ir.UnaryExpr(op=node.op, expr=self.visit(node.expr))

    def visit_BinaryExpr(self, node: gtfn_ir.BinaryExpr):
        return gtfn_ir.BinaryExpr(op=node.op, lhs=self.visit(node.lhs), rhs=self.visit(node.rhs))

    @staticmethod
    def _depth(node: gtfn_ir.FunCall) -> int:
        # TODO bad hardcoded tring
        return (
            1 + ToImpIR._depth(node.args[0])
            if isinstance(node.args[0], gtfn_ir.FunCall) and "step" in node.args[0].fun.id
            else 0
        )

    @staticmethod
    def _peek_init(node: gtfn_ir.FunCall) -> int:
        # TODO bad hardcoded tring
        return (
            ToImpIR._peek_init(node.args[0])
            if isinstance(node.args[0], gtfn_ir.FunCall) and "step" in node.args[0].fun.id
            else node.args[0]
        )

    def visit_Lambda(self, node: gtfn_ir.Lambda, **kwargs):
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
            rhs = self.visit(new_expr)  # TODO: this only supports sum_over
            self.imp_list_ir.append(
                AssignStmt(op="+=", lhs=gtfn_ir.SymRef(id=kwargs["red_idx"]), rhs=rhs)
            )

    def visit_FunCall(self, node: gtfn_ir.FunCall):
        if (
            isinstance(node.fun, gtfn_ir.Lambda) and "step" in node.fun.params[0].id
        ):  # TODO: bad hardcoded string
            #       maybe this could be improved by looking for lambdas that eval their arg or something?
            red_idx = self.uids.sequential_id(prefix="red")
            init = ToImpIR._peek_init(node.fun.expr)
            self.imp_list_ir.append(
                InitStmt(lhs=gtfn_ir.Sym(id=f"{red_idx}"), rhs=self.visit(init))
            )
            num_iter = 1 + ToImpIR._depth(node.fun.expr)
            self.visit(node.args[0], num_iter=num_iter, red_idx=red_idx)
            return gtfn_ir.SymRef(id=f"{red_idx}")
        if isinstance(node.fun, gtfn_ir.Lambda):
            lam_idx = self.uids.sequential_id(prefix="lam")
            params = [self.visit(param) for param in node.fun.params]
            args = [self.visit(arg) for arg in node.args]
            for param, arg in zip(params, args):
                self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{param.id}"), rhs=arg))
            expr = self.visit(node.fun.expr)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{lam_idx}"), rhs=expr))
            return gtfn_ir.SymRef(id=f"{lam_idx}")
        if (
            isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id == "make_tuple"
        ):  # TODO: bad hardcoded string
            tupl_idx = self.uids.sequential_id(prefix="tupl")
            for i, arg in enumerate(node.args):
                expr = self.visit(arg)
                self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{tupl_idx}_{i}"), rhs=expr))
            tup_args = [gtfn_ir.SymRef(id=f"{tupl_idx}_{i}") for i in range(len(node.args))]
            tuple_fun = gtfn_ir.FunCall(fun=gtfn_ir.SymRef(id="make_tuple"), args=tup_args)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{tupl_idx}"), rhs=tuple_fun))
            return gtfn_ir.SymRef(id=f"{tupl_idx}")
        return gtfn_ir.FunCall(
            fun=self.visit(node.fun), args=[self.visit(arg) for arg in node.args]
        )

    def visit_TernaryExpr(self, node: gtfn_ir.TernaryExpr) -> str:
        cond = self.visit(node.cond)
        if_ = self.visit(node.true_expr)
        else_ = self.visit(node.false_expr)
        cond_idx = self.uids.sequential_id(prefix="cond")
        self.imp_list_ir.append(
            Conditional(
                type=f"{cond_idx}_t",
                init_stmt=InitStmt(
                    type=f"{cond_idx}_t",
                    lhs=gtfn_ir.Sym(id=cond_idx),
                    rhs=gtfn_ir.Literal(value="0.", type="float64"),
                ),
                cond=cond,
                if_stmt=AssignStmt(lhs=gtfn_ir.SymRef(id=cond_idx), rhs=if_),
                else_stmt=AssignStmt(lhs=gtfn_ir.SymRef(id=cond_idx), rhs=else_),
            )
        )
        return gtfn_ir.SymRef(id=cond_idx)


@dataclasses.dataclass(frozen=True)
class GTFN_IM_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    def visit_FunctionDefinition(
        self, node: gtfn_ir.FunctionDefinition, **kwargs: Any
    ) -> ImperativeFunctionDefinition:
        to_imp_ir = ToImpIR(imp_list_ir=[])
        ret = to_imp_ir.visit(node.expr)
        return ImperativeFunctionDefinition(
            id=node.id,
            params=node.params,
            fun=to_imp_ir.imp_list_ir + [ReturnStmt(ret=ret)],
        )
