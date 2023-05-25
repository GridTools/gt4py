# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import functools
import operator

from gt4py.eve import NodeTranslator, NodeVisitor, SymbolTableTrait, VisitorWithSymbolTableTrait
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


@dataclasses.dataclass
class _NodeReplacer(NodeTranslator):
    expr_map: dict[int, ir.SymRef]

    def visit_Expr(self, node):
        if id(node) in self.expr_map:
            return self.expr_map[id(node)]
        return self.generic_visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        node = self.visit_Expr(node)
        # If we encounter an expression like:
        #  (λ(_cs_1) → (λ(a) → a+a)(_cs_1))(outer_expr)
        # (non-recursively) inline the lambda to obtain:
        #  (λ(_cs_1) → _cs_1+_cs_1)(outer_expr)
        # This allows identifying more common subexpressions later on
        if isinstance(node, ir.FunCall) and isinstance(node.fun, ir.Lambda):
            eligible_params = []
            for arg in node.args:
                eligible_params.append(isinstance(arg, ir.SymRef) and arg.id.startswith("_cs"))
            if any(eligible_params):
                # note: the inline is opcount preserving anyway so avoid the additional
                # effort in the inliner by disabling opcount preservation.
                return inline_lambda(
                    node, opcount_preserving=False, eligible_params=eligible_params
                )
        return node


def _is_collectable_expr(node: ir.Node):
    if isinstance(node, ir.FunCall):
        # do not collect (and thus deduplicate in CSE) shift(offsets…) calls. Node must still be
        #  visited, to ensure symbol dependencies are recognized correctly.
        # do also not collect reduce nodes if they are left in the it at this point, this may lead to
        #  conceptual problems (other parts of the tool chain rely on the arguments being present directly
        #  on the reduce FunCall node (connectivity deduction)), as well as problems with the imperative backend
        #  backend (single pass eager depth first visit approach)
        if isinstance(node.fun, ir.SymRef) and node.fun.id in ["lift", "shift", "reduce"]:
            return False
        return True
    elif isinstance(node, ir.Lambda):
        return True

    return False


@dataclasses.dataclass
class CollectSubexpressions(VisitorWithSymbolTableTrait, NodeVisitor):
    @dataclasses.dataclass
    class State:
        #: A dictionary mapping a node to a list of node ids which are equal. Additionally, for
        #: each (actual) node we store a set of all ids of collected child subexpressions.
        subexprs: dict[ir.Node, list[tuple[int, set[int]]]] = dataclasses.field(
            default_factory=dict
        )
        # TODO(tehrengruber): Revisit if this makes sense or if we can just recompute the collected
        #  child node ids and get simpler code.
        #: The ids of all child subexpressions which are collected.
        collected_child_node_ids: set[int] = dataclasses.field(default_factory=set)
        #: The ids of all nodes declaring a symbol which are referenced (using a `SymRef`)
        used_symbol_ids: set[int] = dataclasses.field(default_factory=set)

        def remove_subexprs(self, nodes: list[ir.Node]):
            node_ids = {node_id for node in nodes for node_id, _ in self.subexprs.get(node, [])}
            for node in nodes:
                del self.subexprs[node]
            for subexpr_data in self.subexprs.values():
                for _, collected_child_node_ids in subexpr_data:
                    collected_child_node_ids -= node_ids

    @classmethod
    def apply(cls, node: ir.Node):
        state = cls.State()
        obj = cls()
        obj.visit(node, state=state)
        # return subexpression in pre-order of the tree, i.e. the nodes closer to the root come
        # first, and skip the root node itself
        return {k: v for k, v in reversed(state.subexprs.items()) if k is not node}

    def visit(self, node, **kwargs):
        if not isinstance(node, SymbolTableTrait) and not _is_collectable_expr(node):
            return super().visit(node, **kwargs)

        parent_state = kwargs.pop("state")
        collected_child_node_ids = set[int]()
        used_symbol_ids = set[int]()

        # Special handling of `if_(condition, true_branch, false_branch)` like expressions that
        # avoids extracting subexpressions unless they are used in at least two of the three
        # arguments.
        if isinstance(node, ir.FunCall) and node.fun == ir.SymRef(id="if_"):
            assert len(node.args) == 3
            # collect subexpressions for all arguments to the `if_`
            arg_states = [self.State() for _ in node.args]
            for arg, state in zip(node.args, arg_states):
                self.visit(arg, state=state, **kwargs)

            # for each subexpression find in how many of the three arguments they occur
            subexpr_count = {}
            for arg_state in arg_states:
                for subexpr in arg_state.subexprs.keys():
                    subexpr_count.setdefault(subexpr, 0)
                    subexpr_count[subexpr] += 1

            # remove all subexpressions that are not eligible for collection
            eligible_subexprs = {node.args[0].args[0]} | {
                subexpr for subexpr, count in subexpr_count.items() if count >= 2
            }
            for arg_state in arg_states:
                arg_state.remove_subexprs(arg_state.subexprs.keys() - eligible_subexprs)

            # merge the states of the three arguments
            subexprs: dict[ir.Node, list[tuple[int, set[int]]]] = {}
            for state in arg_states:
                for subexpr, data in state.subexprs.items():
                    subexprs.setdefault(subexpr, []).extend(data)
            collected_child_node_ids = functools.reduce(
                operator.or_, (state.collected_child_node_ids for state in arg_states)
            )
            used_symbol_ids = functools.reduce(
                operator.or_, (state.used_symbol_ids for state in arg_states)
            )
            # propagate collected subexpressions to parent
            for subexpr, data in subexprs.items():
                parent_state.subexprs.setdefault(subexpr, []).extend(data)
        else:
            super().visit(
                node,
                state=self.State(parent_state.subexprs, collected_child_node_ids, used_symbol_ids),
                **kwargs,
            )

        if isinstance(node, SymbolTableTrait):
            # remove symbols used in child nodes if they are declared in the current node
            used_symbol_ids = used_symbol_ids - {id(v) for v in node.annex.symtable.values()}

        # if no symbols are used that are defined in the root node, i.e. the node given to `apply`,
        # we collect the subexpression
        if not used_symbol_ids and _is_collectable_expr(node):
            parent_state.subexprs.setdefault(node, []).append((id(node), collected_child_node_ids))

            # propagate to parent that we have collected its child
            parent_state.collected_child_node_ids.add(id(node))

        # propagate used symbol ids to parent
        parent_state.used_symbol_ids.update(used_symbol_ids)

        # propagate to parent which of its children we have collected
        # TODO(tehrengruber): This is expensive for a large tree. Use something like a "ChainSet".
        parent_state.collected_child_node_ids.update(collected_child_node_ids)

    def visit_SymRef(self, node: ir.SymRef, *, symtable, state, **kwargs):
        if node.id in symtable:  # root symbol otherwise
            state.used_symbol_ids.add(id(symtable[node.id]))


@dataclasses.dataclass(frozen=True)
class CommonSubexpressionElimination(NodeTranslator):
    """
    Perform common subexpression elimination.

    Examples:
        >>> x = ir.SymRef(id="x")
        >>> plus = lambda a, b: ir.FunCall(fun=ir.SymRef(id=("plus")), args=[a, b])
        >>> expr = plus(plus(x, x), plus(x, x))
        >>> print(CommonSubexpressionElimination().visit(expr))
        (λ(_cs_1) → _cs_1 + _cs_1)(x + x)
    """

    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    collect_all: bool = dataclasses.field(default=False)

    def visit_FunCall(self, node: ir.FunCall):
        if isinstance(node.fun, ir.SymRef) and node.fun.id in [
            "cartesian_domain",
            "unstructured_domain",
        ]:
            return node

        revisit_node = False

        # collect expressions
        subexprs = CollectSubexpressions.apply(node)

        # collect multiple occurrences and map them to fresh symbols
        expr_map = dict[int, ir.SymRef]()
        params = []
        args = []
        ignored_ids = set()
        for expr, subexpr_entry in subexprs.items():
            if len(subexpr_entry) < 2:
                continue

            eligible_ids = set()
            for id_, child_ids in subexpr_entry:
                if id_ in ignored_ids:
                    # if the node id is ignored (because its parent is eliminated), but it occurs
                    # multiple times then we want to visit the final result once more.
                    revisit_node = True
                else:
                    eligible_ids.add(id_)
                    # since the node id is eligible don't eliminate its children
                    ignored_ids.update(child_ids)

            # if no node ids are eligible, e.g. because the parent was already eliminated or because
            # the expression occurs only once, skip the elimination
            if not eligible_ids:
                continue

            expr_id = self.uids.sequential_id(prefix="_cs")
            params.append(ir.Sym(id=expr_id))
            args.append(expr)
            expr_ref = ir.SymRef(id=expr_id)
            for id_ in eligible_ids:
                expr_map[id_] = expr_ref

        if not expr_map:
            return self.generic_visit(node)

        # apply remapping
        result = ir.FunCall(
            fun=ir.Lambda(params=params, expr=_NodeReplacer(expr_map).visit(node)),
            args=args,
        )
        # TODO(tehrengruber): Instead of revisiting we might be able to replace subexpressions
        #  inside of subexpressions directly. This would require a different order of replacement
        #  (from lower to higher level).
        if revisit_node:
            return self.visit(result)

        return self.generic_visit(result)
