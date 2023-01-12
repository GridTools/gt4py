import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Union

import eve
from eve import NodeTranslator, NodeVisitor
from eve.utils import UIDGenerator
from functional.program_processors.codegens.gtfn import gtfn_ir, gtfn_ir_common
from functional.program_processors.codegens.gtfn.gtfn_im_ir import (
    AssignStmt,
    Conditional,
    ImperativeFunctionDefinition,
    InitStmt,
    ReturnStmt,
    Stmt,
)


def _find_connectivity(reduce_args: Iterable[gtfn_ir_common.Expr], offset_provider):
    connectivities = []
    for arg in reduce_args:
        if isinstance(arg, gtfn_ir.FunCall) and arg.fun == gtfn_ir_common.SymRef(id="shift"):
            assert isinstance(arg.args[-1], gtfn_ir.OffsetLiteral), f"{arg.args}"
            connectivities.append(offset_provider[arg.args[-1].value])

    if not connectivities:
        raise RuntimeError("Couldn't detect partial shift in any arguments of reduce.")

    if len({(c.max_neighbors, c.has_skip_values) for c in connectivities}) != 1:
        # The condition for this check is required but not sufficient: the actual neighbor tables could still be incompatible.
        raise RuntimeError("Arguments to reduce have incompatible partial shifts.")
    return connectivities[0]


def _is_reduce(node: gtfn_ir.FunCall):
    return isinstance(node.fun, gtfn_ir.FunCall) and node.fun.fun == gtfn_ir_common.SymRef(
        id="reduce"
    )


# the current algorithm to turn gtfn_ir into a imperative representation is rather naive,
# it visits the gtfn_ir tree eagerly depth first, and it is only doing one pass. This entails
# that some constructs can not be treated. This visitor checks for such constructs, and makes
# the contract on the gtfn_ir for ToImpIR to suceed explicit
class IsImpCompatible(NodeVisitor):
    incompatible_node: Optional[gtfn_ir_common.Expr] = None
    compatible: bool = True

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs):
        if not self.compatible:
            return

        # reductions need at least one argument with a partial shift
        if _is_reduce(node):
            offset_provider = kwargs["offset_provider"]
            assert offset_provider is not None
            try:
                _find_connectivity(node.args, offset_provider)
            except RuntimeError:
                self.compatible = False
                self.incompatible_node = node

        # function calls need to be lambdas, built ins or reduce
        #   this essentailly avoids situations like
        #   (λ(cs) → cs(w))(λ(x) → x)
        #   where w is some field
        if isinstance(node.fun, gtfn_ir_common.SymRef):
            self.compatible = self.compatible and node.fun.id in gtfn_ir.BUILTINS
            if not self.compatible:
                self.incompatible_node = node

        self.generic_visit(node, **kwargs)


@dataclasses.dataclass(frozen=False)
class GTFN_IM_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @staticmethod
    def asfloat(value: str) -> str:
        if "." not in value and "e" not in value and "E" not in value:
            return f"{value}."
        return value

    def visit_SymRef(self, node: gtfn_ir_common.SymRef, **kwargs):
        if "localized_symbols" in kwargs and node.id in kwargs["localized_symbols"]:
            return gtfn_ir_common.SymRef(id=kwargs["localized_symbols"][node.id])
        return node

    @staticmethod
    def _make_shift(offsets: list[gtfn_ir_common.Expr], iterator: gtfn_ir_common.Expr):
        return gtfn_ir.FunCall(
            fun=gtfn_ir.FunCall(fun=gtfn_ir_common.SymRef(id="shift"), args=offsets),
            args=[iterator],
        )

    @staticmethod
    def _make_deref(iterator: gtfn_ir_common.Expr):
        return gtfn_ir.FunCall(fun=gtfn_ir_common.SymRef(id="deref"), args=[iterator])

    @staticmethod
    def _make_dense_acess(shift_call: gtfn_ir.FunCall, nbh_iter: int):
        return gtfn_ir.FunCall(
            fun=gtfn_ir_common.SymRef(id="deref"),
            args=[
                gtfn_ir.FunCall(
                    fun=gtfn_ir_common.SymRef(id="shift"), args=shift_call.args + [nbh_iter]
                )
            ],
        )

    @staticmethod
    def _make_sparse_acess(field_ref: gtfn_ir_common.SymRef, nbh_iter: int):
        return gtfn_ir.FunCall(
            fun=gtfn_ir_common.SymRef(id="tuple_get"),
            args=[
                nbh_iter,
                gtfn_ir.FunCall(fun=gtfn_ir_common.SymRef(id="deref"), args=[field_ref]),
            ],
        )

    def commit_args(self, node: gtfn_ir.FunCall, tmp_id: str, fun_id: str, **kwargs):
        for i, arg in enumerate(node.args):
            expr = self.visit(arg, **kwargs)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir_common.Sym(id=f"{tmp_id}_{i}"), rhs=expr))
        tup_args = [gtfn_ir_common.SymRef(id=f"{tmp_id}_{i}") for i in range(len(node.args))]
        return gtfn_ir.FunCall(fun=gtfn_ir_common.SymRef(id=fun_id), args=tup_args)

    def handle_Reduction(self, node, **kwargs):
        offset_provider = kwargs["offset_provider"]
        assert offset_provider is not None
        connectivity = _find_connectivity(node.args, offset_provider)
        max_neighbors = connectivity.max_neighbors
        fun, init = node.fun.args
        args = node.args
        # do the following transformations to the node arguments
        # dense fields: shift(dense_f, X2Y) -> deref(shift(dense_f, X2Y, nbh_iterator)
        # sparse_fields: sparse_f -> tuple_get(nbh_iterator, deref(sparse_f)))
        new_args = []
        nbh_iter = gtfn_ir_common.SymRef(id="nbh_iter")
        for arg in args:
            if isinstance(arg, gtfn_ir.FunCall) and arg.fun.id == "shift":
                new_args.append(self._make_dense_acess(arg, nbh_iter))
            if isinstance(arg, gtfn_ir_common.SymRef):
                new_args.append(self._make_sparse_acess(arg, nbh_iter))

        param_to_args = dict(zip([param.id for param in fun.params[1:]], new_args))
        acc = fun.params[0]

        class InlineArgs(NodeTranslator):
            def visit_Expr(self, node):
                if hasattr(node, "id") and node.id in param_to_args:
                    return param_to_args[node.id]
                return self.generic_visit(node)

        new_body = InlineArgs().visit(fun.expr)

        red_idx = self.uids.sequential_id(prefix="red")
        red_lit = gtfn_ir_common.Sym(id=f"{red_idx}")
        self.imp_list_ir.append(InitStmt(lhs=red_lit, rhs=self.visit(init, **kwargs)))

        class PlugInCurrentIdx(NodeTranslator):
            def visit_SymRef(self, node):
                if node.id == acc.id:
                    return gtfn_ir_common.SymRef(id=red_idx)
                if node.id == "nbh_iter":
                    return self.cur_idx
                return self.generic_visit(node)

            def __init__(self, cur_idx):
                self.cur_idx = cur_idx

        for i in range(max_neighbors):
            new_expr = PlugInCurrentIdx(cur_idx=gtfn_ir.OffsetLiteral(value=i)).visit(new_body)
            rhs = self.visit(new_expr, **kwargs)
            self.imp_list_ir.append(AssignStmt(lhs=gtfn_ir_common.SymRef(id=red_idx), rhs=rhs))

        return gtfn_ir_common.SymRef(id=red_idx)

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs):
        if any(
            isinstance(
                arg,
                gtfn_ir.Lambda,
            )
            for arg in node.args
        ):
            # do not try to lower constructs that take lambdas as argument to something more readable
            red_idx = self.uids.sequential_id(prefix="lam")
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir_common.Sym(id=f"{red_idx}"), rhs=node))
            return gtfn_ir_common.SymRef(id=f"{red_idx}")
        if isinstance(node.fun, gtfn_ir.Lambda):
            lam_idx = self.uids.sequential_id(prefix="lam")
            params = [self.visit(param, **kwargs) for param in node.fun.params]
            args = [self.visit(arg, **kwargs) for arg in node.args]
            for param, arg in zip(params, args):
                if param.id in self.sym_table:
                    kwargs["localized_symbols"][
                        param.id
                    ] = f"{param.id}_{self.uids.sequential_id()}_local"
                    self.imp_list_ir.append(
                        InitStmt(
                            lhs=gtfn_ir_common.Sym(id=kwargs["localized_symbols"][param.id]),
                            rhs=arg,
                        )
                    )
                else:
                    self.imp_list_ir.append(
                        InitStmt(
                            lhs=gtfn_ir_common.Sym(id=f"{param.id}"),
                            rhs=arg,
                        )
                    )
            expr = self.visit(node.fun.expr, **kwargs)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir_common.Sym(id=f"{lam_idx}"), rhs=expr))
            return gtfn_ir_common.SymRef(id=f"{lam_idx}")
        if _is_reduce(node):
            return self.handle_Reduction(node, **kwargs)
        if isinstance(node.fun, gtfn_ir_common.SymRef) and node.fun.id == "make_tuple":
            tupl_id = self.uids.sequential_id(prefix="tupl")
            tuple_fun = self.commit_args(node, tupl_id, "make_tuple", **kwargs)
            self.imp_list_ir.append(
                InitStmt(lhs=gtfn_ir_common.Sym(id=f"{tupl_id}"), rhs=tuple_fun)
            )
            return gtfn_ir_common.SymRef(id=f"{tupl_id}")
        return gtfn_ir.FunCall(
            fun=self.visit(node.fun, **kwargs),
            args=[self.visit(arg, **kwargs) for arg in node.args],
        )

    def visit_TernaryExpr(self, node: gtfn_ir.TernaryExpr, **kwargs):
        cond = self.visit(node.cond, **kwargs)
        if_ = self.visit(node.true_expr, **kwargs)
        else_ = self.visit(node.false_expr, **kwargs)
        cond_idx = self.uids.sequential_id(prefix="cond")
        self.imp_list_ir.append(
            Conditional(
                cond_type=f"{cond_idx}_t",
                init_stmt=InitStmt(
                    init_type=f"{cond_idx}_t",
                    lhs=gtfn_ir_common.Sym(id=cond_idx),
                    rhs=gtfn_ir.Literal(value="0.", type="float64"),
                ),
                cond=cond,
                if_stmt=AssignStmt(lhs=gtfn_ir_common.SymRef(id=cond_idx), rhs=if_),
                else_stmt=AssignStmt(lhs=gtfn_ir_common.SymRef(id=cond_idx), rhs=else_),
            )
        )
        return gtfn_ir_common.SymRef(id=cond_idx)

    def visit_FunctionDefinition(
        self, node: gtfn_ir.FunctionDefinition, **kwargs: Any
    ) -> ImperativeFunctionDefinition:
        check_compat = IsImpCompatible()
        check_compat.visit(node.expr, **kwargs)
        # if not check_compat.compatible:
        #     raise RuntimeError(f"gtfn_im can not handle {check_compat.incompatible_node}")

        self.imp_list_ir: List[Union[Stmt, Conditional]] = []
        self.sym_table: Dict[gtfn_ir_common.Sym, gtfn_ir_common.SymRef] = node.annex.symtable
        ret = self.visit(node.expr, localized_symbols={}, **kwargs)

        return ImperativeFunctionDefinition(
            id=node.id,
            params=node.params,
            fun=self.imp_list_ir + [ReturnStmt(ret=ret)],
        )
