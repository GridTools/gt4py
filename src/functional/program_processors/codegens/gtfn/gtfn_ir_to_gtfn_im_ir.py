import dataclasses
from functional.program_processors.codegens.gtfn import gtfn_ir
import eve
from eve import NodeVisitor, NodeTranslator
from eve.utils import UIDGenerator
from functional.iterator import ir
from functional.program_processors.codegens.gtfn.gtfn_im_ir import (
    Stmt,
    InitStmt,
    AssignStmt,
    ReturnStmt,
    Conditional,
    ImperativeFunctionDefinition,
)

from typing import List, Union, Any, Iterable


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

    def visit_Node(self, node, **kwargs):
        return node

    def visit_UnaryExpr(self, node: gtfn_ir.UnaryExpr, **kwargs):
        return gtfn_ir.UnaryExpr(op=node.op, expr=self.visit(node.expr, **kwargs))

    def visit_BinaryExpr(self, node: gtfn_ir.BinaryExpr, **kwargs):
        return gtfn_ir.BinaryExpr(
            op=node.op, lhs=self.visit(node.lhs, **kwargs), rhs=self.visit(node.rhs, **kwargs)
        )

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

    def handle_Lambda(self, node: gtfn_ir.Lambda, **kwargs):
        idx_to_replace = node.params[1].id  # find _i_X parameter

        class PlugInCurrentIdx(NodeTranslator):
            def visit_SymRef(self, node):
                if node.id == idx_to_replace:
                    return gtfn_ir.OffsetLiteral(value=self.cur_idx)
                return self.generic_visit(node)

            def __init__(self, cur_idx: int):
                self.cur_idx = cur_idx

        class UndoCSE(NodeTranslator):
            def visit_SymRef(self, node):
                if node.id in expr_map:
                    return self.expr_map[node.id]
                return self.generic_visit(node)

            def __init__(self, expr_map):
                self.expr_map = expr_map

        for lambda_iter in range(kwargs["num_iter"]):
            # potentially, CSE was run on the contents of the redcution
            #   for now, just undo the CSE pass
            if isinstance(node.expr, gtfn_ir.FunCall) and isinstance(node.expr.fun, gtfn_ir.Lambda):
                expr_map = dict(zip([param.id for param in node.expr.fun.params], node.expr.args))
                nary_expr = UndoCSE(expr_map=expr_map).visit(node.expr.fun.expr)
            else:
                nary_expr = node.expr

            # neighbor sum
            if isinstance(nary_expr, gtfn_ir.BinaryExpr):
                new_expr = PlugInCurrentIdx(cur_idx=lambda_iter).visit(nary_expr.rhs)
                rhs = self.visit(new_expr, **kwargs)
                self.imp_list_ir.append(
                    AssignStmt(op="+=", lhs=gtfn_ir.SymRef(id=kwargs["red_idx"]), rhs=rhs)
                )
            # max_over, min_over
            elif isinstance(nary_expr, gtfn_ir.TernaryExpr):
                new_false_expr = PlugInCurrentIdx(cur_idx=lambda_iter).visit(nary_expr.false_expr)
                fun_name = "maximum" if nary_expr.cond.op == ">" else "minimum"
                self.imp_list_ir.append(
                    AssignStmt(
                        lhs=gtfn_ir.SymRef(id=kwargs["red_idx"]),
                        rhs=gtfn_ir.FunCall(
                            fun=gtfn_ir.SymRef(id=fun_name),
                            args=[gtfn_ir.SymRef(id=kwargs["red_idx"]), new_false_expr],
                        ),
                    )
                )
            else:
                raise NotImplementedError(f"unknown reduction type {nary_expr}")

    @staticmethod
    def _find_connectivity(reduce_args: Iterable[gtfn_ir.Expr], offset_provider):
        connectivities = []
        for arg in reduce_args:
            if isinstance(arg, gtfn_ir.FunCall) and arg.fun == gtfn_ir.SymRef(id="shift"):
                assert isinstance(arg.args[-1], gtfn_ir.OffsetLiteral), f"{arg.args}"
                connectivities.append(offset_provider[arg.args[-1].value])

        if not connectivities:
            raise RuntimeError("Couldn't detect partial shift in any arguments of reduce.")

        if len({(c.max_neighbors, c.has_skip_values) for c in connectivities}) != 1:
            # The condition for this check is required but not sufficient: the actual neighbor tables could still be incompatible.
            raise RuntimeError("Arguments to reduce have incompatible partial shifts.")
        return connectivities[0]

    @staticmethod
    def _is_reduce(node: gtfn_ir.FunCall):
        return isinstance(node.fun, gtfn_ir.FunCall) and node.fun.fun == gtfn_ir.SymRef(id="reduce")

    @staticmethod
    def _make_shift(offsets: list[gtfn_ir.Expr], iterator: gtfn_ir.Expr):
        return gtfn_ir.FunCall(
            fun=gtfn_ir.FunCall(fun=gtfn_ir.SymRef(id="shift"), args=offsets), args=[iterator]
        )

    @staticmethod
    def _make_deref(iterator: gtfn_ir.Expr):
        return gtfn_ir.FunCall(fun=gtfn_ir.SymRef(id="deref"), args=[iterator])

    def handle_Reduction(self, node, **kwargs):
        offset_provider = kwargs["offset_provider"]
        assert offset_provider is not None
        connectivity = self._find_connectivity(node.args, offset_provider)
        max_neighbors = connectivity.max_neighbors
        fun, init = node.fun.args
        args = node.args
        # do the following transformations to the node arguments
        # dense fields: shift(dense_f, X2Y) -> deref(shift(dense_f, X2Y, nbh_iterator)
        # sparse_fields: sparse_f -> tuple_get(nbh_iterator, deref(sparse_f)))
        new_args = []
        nbh_iter = gtfn_ir.SymRef(id="nbh_iter")
        for arg in args:
            if isinstance(arg, gtfn_ir.FunCall) and arg.fun.id == "shift":
                new_args.append(
                    gtfn_ir.FunCall(
                        fun=gtfn_ir.SymRef(id="deref"),
                        args=[
                            gtfn_ir.FunCall(
                                fun=gtfn_ir.SymRef(id="shift"), args=arg.args + [nbh_iter]
                            )
                        ],
                    )
                )
            if isinstance(arg, gtfn_ir.SymRef):
                new_args.append(
                    gtfn_ir.FunCall(
                        fun=gtfn_ir.SymRef(id="tuple_get"),
                        args=[
                            nbh_iter,
                            gtfn_ir.FunCall(fun=gtfn_ir.SymRef(id="deref"), args=[arg]),
                        ],
                    )
                )

        old_to_new_args = dict(zip([param.id for param in fun.params[1:]], new_args))
        acc = fun.params[0]

        class ReplaceArgs(NodeTranslator):
            def visit_Expr(self, node):
                if hasattr(node, "id") and node.id in old_to_new_args:
                    return old_to_new_args[node.id]
                return self.generic_visit(node)

        new_fun = ReplaceArgs().visit(fun.expr)

        red_idx = self.uids.sequential_id(prefix="red")
        red_lit = gtfn_ir.Sym(id=f"{red_idx}")
        self.imp_list_ir.append(InitStmt(lhs=red_lit, rhs=self.visit(init, **kwargs)))

        class PlugInCurrentIdx(NodeTranslator):
            def visit_SymRef(self, node):
                if node.id == acc.id:
                    return gtfn_ir.SymRef(id=red_idx)
                if node.id == "nbh_iter":
                    return gtfn_ir.OffsetLiteral(value=self.cur_idx)
                return self.generic_visit(node)

            def __init__(self, cur_idx: int):
                self.cur_idx = cur_idx

        for i in range(max_neighbors):
            new_expr = PlugInCurrentIdx(cur_idx=i).visit(new_fun)
            rhs = self.visit(new_expr, **kwargs)
            self.imp_list_ir.append(AssignStmt(lhs=gtfn_ir.SymRef(id=red_idx), rhs=rhs))

        return gtfn_ir.SymRef(id=red_idx)

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs):
        if isinstance(node.fun, gtfn_ir.Lambda) and any(
            isinstance(
                arg,
                gtfn_ir.Lambda,
            )
            for arg in node.args
        ):
            # do not try to lower lambdas that take lambdas as arugment to something more readable
            red_idx = self.uids.sequential_id(prefix="red")
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{red_idx}"), rhs=node))
            return gtfn_ir.SymRef(id=f"{red_idx}")
        if isinstance(node.fun, gtfn_ir.Lambda):
            lam_idx = self.uids.sequential_id(prefix="lam")
            params = [self.visit(param, **kwargs) for param in node.fun.params]
            args = [self.visit(arg, **kwargs) for arg in node.args]
            for param, arg in zip(params, args):
                self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{param.id}"), rhs=arg))
            expr = self.visit(node.fun.expr, **kwargs)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{lam_idx}"), rhs=expr))
            return gtfn_ir.SymRef(id=f"{lam_idx}")
        if self._is_reduce(node):
            return self.handle_Reduction(node, **kwargs)
        if (
            isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id == "make_tuple"
        ):  # TODO: bad hardcoded string
            tupl_idx = self.uids.sequential_id(prefix="tupl")
            for i, arg in enumerate(node.args):
                expr = self.visit(arg, **kwargs)
                self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{tupl_idx}_{i}"), rhs=expr))
            tup_args = [gtfn_ir.SymRef(id=f"{tupl_idx}_{i}") for i in range(len(node.args))]
            tuple_fun = gtfn_ir.FunCall(fun=gtfn_ir.SymRef(id="make_tuple"), args=tup_args)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir.Sym(id=f"{tupl_idx}"), rhs=tuple_fun))
            return gtfn_ir.SymRef(id=f"{tupl_idx}")
        return gtfn_ir.FunCall(
            fun=self.visit(node.fun, **kwargs),
            args=[self.visit(arg, **kwargs) for arg in node.args],
        )

    def visit_TernaryExpr(self, node: gtfn_ir.TernaryExpr, **kwargs) -> str:
        cond = self.visit(node.cond, **kwargs)
        if_ = self.visit(node.true_expr, **kwargs)
        else_ = self.visit(node.false_expr, **kwargs)
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
        ret = to_imp_ir.visit(node.expr, **kwargs)
        return ImperativeFunctionDefinition(
            id=node.id,
            params=node.params,
            fun=to_imp_ir.imp_list_ir + [ReturnStmt(ret=ret)],
        )
