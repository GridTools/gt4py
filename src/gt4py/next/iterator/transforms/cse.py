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
import typing

from gt4py.eve import NodeTranslator, NodeVisitor, SymbolTableTrait, VisitorWithSymbolTableTrait
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


@dataclasses.dataclass
class _NodeReplacer(NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ["type"]

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


def _is_if_can_deref(node: ir.Node):
    # `if_(can_deref(...), ..., ...)`
    return (
        isinstance(node, ir.FunCall)
        and node.fun == ir.SymRef(id="if_")
        and isinstance(node.args[0], ir.FunCall)
        and node.args[0].fun == ir.SymRef(id="can_deref")
    )


@dataclasses.dataclass
class CollectSubexpressions(VisitorWithSymbolTableTrait, NodeVisitor):
    subexprs: dict[ir.Node, list[tuple[int, set[int]]]] = dataclasses.field(
        init=False, repr=False, default_factory=dict
    )

    @classmethod
    def apply(cls, node: ir.Node):
        obj = cls()
        obj.visit(
            node, used_symbol_ids=set(), collected_child_node_ids=set(), allow_collection=True
        )
        # return subexpression in pre-order of the tree, i.e. the nodes closer to the root come
        # first, and skip the root node itself
        return {k: v for k, v in reversed(obj.subexprs.items()) if k is not node}

    def visit(self, node, **kwargs):
        # TODO(tehrengruber): improve this case as we might miss subexpression that could be eliminated
        # disable collection (for all child nodes) if node matches `if_(can_deref(...), ..., ...)`
        if _is_if_can_deref(node):
            kwargs["allow_collection"] = False

        if not isinstance(node, SymbolTableTrait) and not _is_collectable_expr(node):
            return super().visit(node, **kwargs)

        parents_collected_child_node_ids = kwargs.pop("collected_child_node_ids")
        parents_used_symbol_ids = kwargs.pop("used_symbol_ids")
        used_symbol_ids = set[int]()
        collected_child_node_ids = set[int]()
        super().visit(
            node,
            used_symbol_ids=used_symbol_ids,
            collected_child_node_ids=collected_child_node_ids,
            **kwargs,
        )

        if isinstance(node, SymbolTableTrait):
            # remove symbols used in child nodes if they are declared in the current node
            used_symbol_ids = used_symbol_ids - {id(v) for v in node.annex.symtable.values()}

        # if no symbols are used that are defined in the root node, i.e. the node given to `apply`,
        # we collect the subexpression
        if not used_symbol_ids and _is_collectable_expr(node) and kwargs["allow_collection"]:
            self.subexprs.setdefault(node, []).append((id(node), collected_child_node_ids))

            # propagate to parent that we have collected its child
            parents_collected_child_node_ids.add(id(node))

        # propagate used symbol ids to parent
        parents_used_symbol_ids.update(used_symbol_ids)

        # propagate to parent which of its children we have collected
        # TODO(tehrengruber): This is expensive for a large tree. Use something like a "ChainSet".
        parents_collected_child_node_ids.update(collected_child_node_ids)

    def visit_SymRef(self, node: ir.SymRef, *, symtable, used_symbol_ids, **kwargs):
        if node.id in symtable:  # root symbol otherwise
            used_symbol_ids.add(id(symtable[node.id]))


def extract_subexpression(
    node: ir.Expr,
    predicate: typing.Callable[[ir.Expr, int], bool],
    uid_generator: UIDGenerator,
    once_only: bool = False,
    pre_order: bool = True
) -> tuple[ir.Expr, dict[ir.Sym, ir.Expr], bool]:
    """
    Given an expression extract all subexprs and return a new expr with the subexprs replaced.

    The return value is a triplet of
    - the new expr with all extracted subexpressions replaced by a reference to a new symbol
    - a dictionary mapping each new symbol to the respective subexpr that was extracted
    - a boolean indicating if a subexression was not collected because its parent was already
      collected.

     Arguments:
        node: The node to extract from.
        predicate: If this predicate evaluates to true the respective subexpression is extracted.
          Takes a subexpression and the number of occurences of the subexpression in the root node
          as arguments.
        uid_generator: The uid generator used to generate new symbol names.
        once_only: If set extraction is stopped after the first expression that is extracted
        pre_order: Extract in pre- or post-order of the root node.


    Examples:
        >>> import gt4py.next.iterator.ir_makers as im
        >>> from gt4py.eve.utils import UIDGenerator
        >>> expr = im.plus(im.plus("x", "y"), im.plus(im.plus("x", "y"), "z"))
        >>> predicate = lambda subexpr, num_occurences: num_occurences > 1
        >>> new_expr, extracted_subexprs, _ = extract_subexpression(
        ...                                     expr, predicate, UIDGenerator(prefix="_subexpr"))
        >>> print(new_expr)
        _subexpr_1 + (_subexpr_1 + z)
        >>> for sym, subexpr in extracted_subexprs.items():
        ...    print(f"`{sym}`: `{subexpr}`")
        `_subexpr_1`: `x + y`
    """
    ignored_children = False
    extracted = dict[ir.Sym, ir.Expr]()

    # collect expressions
    subexprs = CollectSubexpressions.apply(node)

    # collect multiple occurrences and map them to fresh symbols
    expr_map = dict[int, ir.SymRef]()
    ignored_ids = set()
    for expr, subexpr_entry in (subexprs.items() if pre_order else reversed(subexprs.items())):
        if not predicate(expr, len(subexpr_entry)):
            continue

        eligible_ids = set()
        for id_, child_ids in subexpr_entry:
            if id_ in ignored_ids:
                ignored_children = True
            else:
                eligible_ids.add(id_)
                # since the node id is eligible don't eliminate its children
                ignored_ids.update(child_ids)

        # if no node ids are eligible, e.g. because the parent was already eliminated or because
        # the expression occurs only once, skip the elimination
        if not eligible_ids:
            continue

        expr_id = uid_generator.sequential_id()
        extracted[ir.Sym(id=expr_id)] = expr
        expr_ref = ir.SymRef(id=expr_id)
        for id_ in eligible_ids:
            expr_map[id_] = expr_ref

        if once_only:
            break

    if not expr_map:
        return node, None, False

    return _NodeReplacer(expr_map).visit(node), extracted, ignored_children



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
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=lambda: UIDGenerator(prefix="_cs"))

    collect_all: bool = dataclasses.field(default=False)

    def visit_FunCall(self, node: ir.FunCall):
        if isinstance(node.fun, ir.SymRef) and node.fun.id in [
            "cartesian_domain",
            "unstructured_domain",
        ]:
            return node

        new_expr, extracted, ignored_children = extract_subexpression(
            node,
            lambda subexpr, num_occurences: num_occurences > 1,
            self.uids
        )

        if not extracted:
            return self.generic_visit(node)

        # apply remapping
        result = ir.FunCall(
            fun=ir.Lambda(params=list(extracted.keys()), expr=new_expr),
            args=list(extracted.values()),
        )

        # if the node id is ignored (because its parent is eliminated), but it occurs
        # multiple times then we want to visit the final result once more.
        # TODO(tehrengruber): Instead of revisiting we might be able to replace subexpressions
        #  inside of subexpressions directly. This would require a different order of replacement
        #  (from lower to higher level).
        if ignored_children:
            return self.visit(result)

        return self.generic_visit(result)
