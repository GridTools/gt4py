# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Dict, Iterable, Iterator, List, Optional, TypeGuard, Union

import gt4py.eve as eve
from gt4py.eve import NodeTranslator, concepts
from gt4py.eve.utils import UIDGenerator
from gt4py.next.program_processors.codegens.gtfn import gtfn_ir, gtfn_ir_common
from gt4py.next.program_processors.codegens.gtfn.gtfn_im_ir import (
    AssignStmt,
    Conditional,
    EmptyListInitializer,
    ImperativeFunctionDefinition,
    InitStmt,
    ReturnStmt,
    Stmt,
)


# TODO: start of code clone from unroll_reduce.py. This is necessary since whilet the IR nodes are compatible between itir and gtfn_ir,
#       the structure of the ir is slightly different, hence functions like _is_shifted and _get_partial_offset_tag are slightly changed
#       in this version of the code clone. To be removed asap
def _is_shifted(arg: gtfn_ir_common.Expr) -> TypeGuard[gtfn_ir.FunCall]:
    return (
        isinstance(arg, gtfn_ir.FunCall)
        and isinstance(arg.fun, gtfn_ir_common.SymRef)
        and arg.fun.id == "shift"
    )


def _is_applied_lift(arg: gtfn_ir_common.Expr) -> TypeGuard[gtfn_ir.FunCall]:
    return (
        isinstance(arg, gtfn_ir.FunCall)
        and isinstance(arg.fun, gtfn_ir.FunCall)
        and arg.fun.fun == gtfn_ir_common.SymRef(id="lift")
    )


def _is_shifted_or_lifted_and_shifted(arg: gtfn_ir_common.Expr) -> TypeGuard[gtfn_ir.FunCall]:
    return _is_shifted(arg) or (
        _is_applied_lift(arg)
        and any(_is_shifted_or_lifted_and_shifted(nested_arg) for nested_arg in arg.args)
    )


def _get_shifted_args(reduce_args: Iterable[gtfn_ir_common.Expr]) -> Iterator[gtfn_ir.FunCall]:
    return filter(_is_shifted_or_lifted_and_shifted, reduce_args)


def _is_list_of_funcalls(lst: list) -> TypeGuard[list[gtfn_ir.FunCall]]:
    return all(isinstance(f, gtfn_ir.FunCall) for f in lst)


def _get_partial_offset_tag(arg: gtfn_ir.FunCall) -> str:
    if _is_shifted(arg):
        assert isinstance(arg.fun, gtfn_ir_common.SymRef)
        offset = arg.args[-1]
        assert isinstance(offset, gtfn_ir.OffsetLiteral)
        assert isinstance(offset.value, str)
        return offset.value
    else:
        assert _is_applied_lift(arg)
        assert _is_list_of_funcalls(arg.args)
        partial_offsets = [_get_partial_offset_tag(arg) for arg in arg.args]
        assert all(o == partial_offsets[0] for o in partial_offsets)
        return partial_offsets[0]


def _get_partial_offset_tags(reduce_args: Iterable[gtfn_ir_common.Expr]) -> Iterable[str]:
    return [_get_partial_offset_tag(arg) for arg in _get_shifted_args(reduce_args)]


def _is_reduce(node: gtfn_ir.FunCall) -> TypeGuard[gtfn_ir.FunCall]:
    return isinstance(node.fun, gtfn_ir.FunCall) and node.fun.fun == gtfn_ir_common.SymRef(
        id="reduce"
    )


# TODO: end of code clone


class PlugInCurrentIdx(NodeTranslator):
    def visit_SymRef(
        self, node: gtfn_ir_common.SymRef
    ) -> gtfn_ir.OffsetLiteral | gtfn_ir_common.SymRef:
        if node.id == "nbh_iter":
            return self.cur_idx
        if self.acc is not None and node.id == self.acc.id:
            return gtfn_ir_common.SymRef(id=self.red_idx)
        return self.generic_visit(node)

    def __init__(
        self, cur_idx: gtfn_ir.OffsetLiteral, acc: Optional[gtfn_ir_common.Sym], red_idx: str
    ):
        self.cur_idx = cur_idx
        self.acc = acc
        self.red_idx = red_idx


@dataclasses.dataclass(frozen=False)
class GTFN_IM_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    def visit_SymRef(self, node: gtfn_ir_common.SymRef, **kwargs: Any) -> gtfn_ir_common.SymRef:
        if "localized_symbols" in kwargs and node.id in kwargs["localized_symbols"]:
            return gtfn_ir_common.SymRef(id=kwargs["localized_symbols"][node.id])
        return node

    def commit_args(
        self, node: gtfn_ir.FunCall, tmp_id: str, fun_id: str, **kwargs: Any
    ) -> gtfn_ir.FunCall:
        for i, arg in enumerate(node.args):
            expr = self.visit(arg, **kwargs)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir_common.Sym(id=f"{tmp_id}_{i}"), rhs=expr))
        tup_args = [gtfn_ir_common.SymRef(id=f"{tmp_id}_{i}") for i in range(len(node.args))]
        return gtfn_ir.FunCall(fun=gtfn_ir_common.SymRef(id=fun_id), args=tup_args)

    def _expand_lambda(
        self,
        node: gtfn_ir.FunCall,
        new_args: List[gtfn_ir.FunCall],
        red_idx: str,
        max_neighbors: int,
        **kwargs: Any,
    ) -> None:
        fun, init = node.fun.args  # type: ignore
        param_to_args = dict(zip([param.id for param in fun.params[1:]], new_args))
        acc = fun.params[0]

        class InlineArgs(NodeTranslator):
            def visit_Expr(self, node: gtfn_ir_common.Expr) -> gtfn_ir_common.Expr:
                if hasattr(node, "id") and node.id in param_to_args:
                    return param_to_args[node.id]
                return self.generic_visit(node)

        new_body = InlineArgs().visit(fun.expr)

        red_lit = gtfn_ir_common.Sym(id=f"{red_idx}")
        self.imp_list_ir.append(InitStmt(lhs=red_lit, rhs=self.visit(init, **kwargs)))

        for i in range(max_neighbors):
            new_expr = PlugInCurrentIdx(
                cur_idx=gtfn_ir.OffsetLiteral(value=i), acc=acc, red_idx=red_idx
            ).visit(new_body)
            rhs = self.visit(new_expr, **kwargs)
            self.imp_list_ir.append(AssignStmt(lhs=gtfn_ir_common.SymRef(id=red_idx), rhs=rhs))

    def _expand_symref(
        self,
        node: gtfn_ir.FunCall,
        new_args: List[gtfn_ir.FunCall],
        red_idx: str,
        max_neighbors: int,
        **kwargs: Any,
    ) -> None:
        fun, init = node.fun.args  # type: ignore

        red_lit = gtfn_ir_common.Sym(id=f"{red_idx}")
        self.imp_list_ir.append(InitStmt(lhs=red_lit, rhs=self.visit(init, **kwargs)))

        for i in range(max_neighbors):
            plugged_in_args = [
                PlugInCurrentIdx(
                    cur_idx=gtfn_ir.OffsetLiteral(value=i), acc=None, red_idx=red_idx
                ).visit(arg)
                for arg in new_args
            ]
            rhs = gtfn_ir.FunCall(
                fun=fun, args=[gtfn_ir_common.SymRef(id=red_idx), *plugged_in_args]
            )
            self.imp_list_ir.append(AssignStmt(lhs=gtfn_ir_common.SymRef(id=red_idx), rhs=rhs))

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs: Any) -> gtfn_ir_common.Expr:
        if any(isinstance(arg, gtfn_ir.Lambda) for arg in node.args):
            # do not try to lower constructs that take lambdas as argument to something more readable
            lam_idx = self.uids.sequential_id(prefix="lam")
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir_common.Sym(id=f"{lam_idx}"), rhs=node))
            return gtfn_ir_common.SymRef(id=f"{lam_idx}")
        if isinstance(node.fun, gtfn_ir.Lambda):
            localized_symbols = {**kwargs["localized_symbols"]}  # create a new scope

            lam_idx = self.uids.sequential_id(prefix="lam")
            params = [self.visit(param, **kwargs) for param in node.fun.params]
            args = [self.visit(arg, **kwargs) for arg in node.args]
            for param, arg in zip(params, args):
                localized_symbols[param.id] = new_symbol = (
                    f"{param.id}_{self.uids.sequential_id()}_local"
                )
                self.imp_list_ir.append(
                    InitStmt(
                        lhs=gtfn_ir_common.Sym(id=new_symbol),
                        rhs=arg,
                    )
                )
            new_kwargs = {**kwargs, "localized_symbols": localized_symbols}
            expr = self.visit(node.fun.expr, **new_kwargs)
            self.imp_list_ir.append(InitStmt(lhs=gtfn_ir_common.Sym(id=f"{lam_idx}"), rhs=expr))
            return gtfn_ir_common.SymRef(id=f"{lam_idx}")
        if _is_reduce(node):
            raise AssertionError(
                "Not implemented. The code-path was removed as it was not actively used and tested."
            )
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

    def visit_TernaryExpr(self, node: gtfn_ir.TernaryExpr, **kwargs: Any) -> gtfn_ir_common.SymRef:
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
                    rhs=EmptyListInitializer(),
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
        self.imp_list_ir: List[Union[Stmt, Conditional]] = []
        self.sym_table: Dict[gtfn_ir_common.Sym, gtfn_ir_common.SymRef] = node.annex.symtable
        ret = self.visit(node.expr, localized_symbols={}, **kwargs)

        return ImperativeFunctionDefinition(
            id=node.id, params=node.params, fun=[*self.imp_list_ir, ReturnStmt(ret=ret)]
        )

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        kwargs = {
            **kwargs,
            "inside_fun": kwargs.get("inside_fun", False)
            or isinstance(node, gtfn_ir.FunctionDefinition),
        }

        # only transform things inside of function definitions and leave the rest as is
        if kwargs["inside_fun"]:
            # we are inside a function, visit as usual
            return super().visit(node, **kwargs)
        else:
            # we are outside of a function, don't transform anything (i.e. don't call any `vist_`
            # methods on `node`), but just visit child nodes.
            return self.generic_visit(node, **kwargs)

    def visit_ScanPassDefinition(
        self, node: gtfn_ir.ScanPassDefinition, **kwargs: Any
    ) -> gtfn_ir.ScanPassDefinition:
        return node

    def visit_ScanExecution(
        self, node: gtfn_ir.ScanExecution, **kwargs: Any
    ) -> gtfn_ir.ScanExecution:
        return node
