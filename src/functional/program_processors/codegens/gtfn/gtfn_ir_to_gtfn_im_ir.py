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
    ForLoop,
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
        emit_for_loop = False

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
                    return self.cur_idx
                return self.generic_visit(node)

            def __init__(self, cur_idx):
                self.cur_idx = cur_idx

        if emit_for_loop:
            new_expr = PlugInCurrentIdx(gtfn_ir.SymRef(id="red_iter")).visit(new_fun)
            rhs = self.visit(new_expr, **kwargs)
            self.imp_list_ir.append(
                ForLoop(
                    num_iter=max_neighbors, stmt=AssignStmt(lhs=gtfn_ir.SymRef(id=red_idx), rhs=rhs)
                )
            )
        else:
            for i in range(max_neighbors):
                new_expr = PlugInCurrentIdx(cur_idx=gtfn_ir.OffsetLiteral(value=i)).visit(new_fun)
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
