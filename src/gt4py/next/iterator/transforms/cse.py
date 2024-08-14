# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import math
import operator
import typing

from gt4py.eve import (
    NodeTranslator,
    NodeVisitor,
    PreserveLocationVisitor,
    SymbolTableTrait,
    VisitorWithSymbolTableTrait,
)
from gt4py.eve.utils import UIDGenerator
from gt4py.next.iterator import ir
from gt4py.next.iterator.transforms.inline_lambdas import inline_lambda


@dataclasses.dataclass
class _NodeReplacer(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("type",)

    expr_map: dict[int, ir.SymRef]

    def visit_Expr(self, node: ir.Node) -> ir.Node:
        if id(node) in self.expr_map:
            return self.expr_map[id(node)]
        return self.generic_visit(node)

    def visit_FunCall(self, node: ir.FunCall) -> ir.Node:
        node = typing.cast(ir.FunCall, self.visit_Expr(node))
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


def _is_collectable_expr(node: ir.Node) -> bool:
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
class CollectSubexpressions(PreserveLocationVisitor, VisitorWithSymbolTableTrait, NodeVisitor):
    @dataclasses.dataclass
    class SubexpressionData:
        #: A list of node ids with equal hash and a set of collected child subexpression ids
        subexprs: list[tuple[int, set[int]]] = dataclasses.field(default_factory=list)
        #: Maximum depth of a subexpression in the tree. Used to sort collected subexpressions
        #:  such that deeper nodes can be processed (in other passed building upon this pass)
        #:  earlier.
        max_depth: int | float = -math.inf

    @dataclasses.dataclass
    class State:
        #: A dictionary mapping a node to a list of node ids with equal hash and some additional
        #: information. See `SubexpressionData` for more information.
        subexprs: dict[ir.Node, "CollectSubexpressions.SubexpressionData"] = dataclasses.field(
            default_factory=dict
        )
        # TODO(tehrengruber): Revisit if this makes sense or if we can just recompute the collected
        #  child node ids and get simpler code.
        #: The ids of all child subexpressions which are collected.
        collected_child_node_ids: set[int] = dataclasses.field(default_factory=set)
        #: The ids of all nodes declaring a symbol which are referenced (using a `SymRef`)
        used_symbol_ids: set[int] = dataclasses.field(default_factory=set)

        def remove_subexprs(self, nodes: typing.Iterable[ir.Node]) -> None:
            node_ids_to_remove: set[int] = set()
            for node in nodes:
                subexpr_data = self.subexprs.pop(node, None)
                if subexpr_data:
                    node_ids, _ = zip(*subexpr_data.subexprs)
                    node_ids_to_remove |= set(node_ids)
            for subexpr_data in self.subexprs.values():
                for _, collected_child_node_ids in subexpr_data.subexprs:
                    collected_child_node_ids -= node_ids_to_remove

    @classmethod
    def apply(cls, node: ir.Node) -> dict[ir.Node, list[tuple[int, set[int]]]]:
        state = cls.State()
        obj = cls()
        obj.visit(node, state=state, depth=-1)
        # Return subexpression such that the nodes closer to the root come first and skip the root
        # node itself.
        subexprs_sorted: list[tuple[ir.Node, "CollectSubexpressions.SubexpressionData"]] = sorted(
            state.subexprs.items(), key=lambda el: el[1].max_depth
        )
        return {k: v.subexprs for k, v in subexprs_sorted if k is not node}

    def generic_visit(self, *args, **kwargs):
        depth = kwargs.pop("depth")
        return super().generic_visit(*args, depth=depth + 1, **kwargs)

    def visit(self, node: ir.Node, **kwargs) -> None:  # type: ignore[override] # supertype accepts any node, but we want to be more specific here.
        if not isinstance(node, SymbolTableTrait) and not _is_collectable_expr(node):
            return super().visit(node, **kwargs)

        depth = kwargs["depth"]
        parent_state = kwargs.pop("state")
        collected_child_node_ids: set[int] = set()
        used_symbol_ids: set[int] = set()

        # Special handling of `if_(condition, true_branch, false_branch)` like expressions that
        # avoids extracting subexpressions unless they are used in either the condition or both
        # branches.
        if isinstance(node, ir.FunCall) and node.fun == ir.SymRef(id="if_"):
            assert len(node.args) == 3
            # collect subexpressions for all arguments to the `if_`
            arg_states = [self.State() for _ in node.args]
            for arg, state in zip(node.args, arg_states):
                self.visit(arg, state=state, **{**kwargs, "depth": depth + 1})

            # remove all subexpressions that are not eligible for collection
            #  (either they occur in the condition or in both branches)
            eligible_subexprs = arg_states[0].subexprs.keys() | (
                arg_states[1].subexprs.keys() & arg_states[2].subexprs.keys()
            )
            for arg_state in arg_states:
                arg_state.remove_subexprs(arg_state.subexprs.keys() - eligible_subexprs)

            # merge the states of the three arguments
            subexprs: dict[ir.Node, CollectSubexpressions.SubexpressionData] = {}
            for state in arg_states:
                for subexpr, data in state.subexprs.items():
                    merged_data = subexprs.setdefault(subexpr, self.SubexpressionData())
                    merged_data.subexprs.extend(data.subexprs)
                    merged_data.max_depth = max(merged_data.max_depth, data.max_depth)
            collected_child_node_ids = functools.reduce(
                operator.or_, (state.collected_child_node_ids for state in arg_states)
            )
            used_symbol_ids = functools.reduce(
                operator.or_, (state.used_symbol_ids for state in arg_states)
            )
            # propagate collected subexpressions to parent
            for subexpr, data in subexprs.items():
                parent_data = parent_state.subexprs.setdefault(subexpr, self.SubexpressionData())
                parent_data.subexprs.extend(data.subexprs)
                parent_data.max_depth = max(merged_data.max_depth, data.max_depth)
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
            parent_data = parent_state.subexprs.setdefault(node, self.SubexpressionData())
            parent_data.subexprs.append((id(node), collected_child_node_ids))
            parent_data.max_depth = max(parent_data.max_depth, depth)

            # propagate to parent that we have collected its child
            parent_state.collected_child_node_ids.add(id(node))

        # propagate used symbol ids to parent
        parent_state.used_symbol_ids.update(used_symbol_ids)

        # propagate to parent which of its children we have collected
        # TODO(tehrengruber): This is expensive for a large tree. Use something like a "ChainSet".
        parent_state.collected_child_node_ids.update(collected_child_node_ids)

    def visit_SymRef(
        self, node: ir.SymRef, *, symtable: dict[str, ir.Node], state: State, **kwargs
    ) -> None:
        if node.id in symtable:  # root symbol otherwise
            state.used_symbol_ids.add(id(symtable[node.id]))


def extract_subexpression(
    node: ir.Expr,
    predicate: typing.Callable[[ir.Expr, int], bool],
    uid_generator: UIDGenerator,
    once_only: bool = False,
    deepest_expr_first: bool = False,
) -> tuple[ir.Expr, typing.Union[dict[ir.Sym, ir.Expr], None], bool]:
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
        once_only: If set extraction is stopped after the first expression that is extracted.
        deepest_expr_first: Extract subexpressions that are lower in the tree first. Otherwise,
          expressions closer to the root are extracted first. Requires `once_only == True`.


    Examples:
        Default case for `(x+y) + ((x+y)+z)`:

        >>> import gt4py.next.iterator.ir_utils.ir_makers as im
        >>> from gt4py.eve.utils import UIDGenerator
        >>> expr = im.plus(im.plus("x", "y"), im.plus(im.plus("x", "y"), "z"))
        >>> predicate = lambda subexpr, num_occurences: num_occurences > 1
        >>> new_expr, extracted_subexprs, _ = extract_subexpression(
        ...     expr, predicate, UIDGenerator(prefix="_subexpr")
        ... )
        >>> print(new_expr)
        _subexpr_1 + (_subexpr_1 + z)
        >>> for sym, subexpr in extracted_subexprs.items():
        ...     print(f"`{sym}`: `{subexpr}`")
        `_subexpr_1`: `x + y`

        The order of the extraction can be configured using `deepest_expr_first`. By default, the nodes
        closer to the root are eliminated first:

        >>> expr = im.plus(
        ...     im.plus(im.plus("x", "y"), im.plus("x", "y")),
        ...     im.plus(im.plus("x", "y"), im.plus("x", "y")),
        ... )
        >>> new_expr, extracted_subexprs, ignored_children = extract_subexpression(
        ...     expr, predicate, UIDGenerator(prefix="_subexpr"), deepest_expr_first=False
        ... )
        >>> print(new_expr)
        _subexpr_1 + _subexpr_1
        >>> for sym, subexpr in extracted_subexprs.items():
        ...     print(f"`{sym}`: `{subexpr}`")
        `_subexpr_1`: `x + y + (x + y)`

        Since `(x+y)` is a child of one of the expressions it is ignored:

        >>> print(ignored_children)
        True

        Setting `deepest_expr_first` will extract nodes deeper in the tree first:

        >>> expr = im.plus(
        ...     im.plus(im.plus("x", "y"), im.plus("x", "y")),
        ...     im.plus(im.plus("x", "y"), im.plus("x", "y")),
        ... )
        >>> new_expr, extracted_subexprs, _ = extract_subexpression(
        ...     expr,
        ...     predicate,
        ...     UIDGenerator(prefix="_subexpr"),
        ...     once_only=True,
        ...     deepest_expr_first=True,
        ... )
        >>> print(new_expr)
        _subexpr_1 + _subexpr_1 + (_subexpr_1 + _subexpr_1)
        >>> for sym, subexpr in extracted_subexprs.items():
        ...     print(f"`{sym}`: `{subexpr}`")
        `_subexpr_1`: `x + y`

        Note that this requires `once_only` to be set right now.
    """
    if deepest_expr_first and not once_only:
        # TODO(tehrengruber): Revisit. We could fix this, but is this case even needed?
        # If we traverse the deepest nodes first, but don't stop after the first extraction the new
        # expression is not meaningful in the current implementation. Just disallow this case for
        # now. E.g.:
        # Input expression:
        #   `((x + y) + (x + y)) + ((x + y) + (x + y))`
        # New expression:
        #   `_subexpr_2 + _subexpr_2`
        # Extracted subexpression:
        #  `_subexpr_1`: `x + y`  (This subexpression is not used anywhere)
        #  `_subexpr_2`: `x + y + (x + y)`
        raise NotImplementedError(
            "Results of the current implementation not meaningful for "
            "'deepest_expr_first == True' and 'once_only == True'."
        )

    ignored_children = False
    extracted = dict[ir.Sym, ir.Expr]()

    # collect expressions
    subexprs = CollectSubexpressions.apply(node)

    # collect multiple occurrences and map them to fresh symbols
    expr_map = dict[int, ir.SymRef]()
    ignored_ids = set()
    for expr, subexpr_entry in (
        subexprs.items() if not deepest_expr_first else reversed(subexprs.items())
    ):
        # just to make mypy happy when calling the predicate. Every subnode and hence subexpression
        # is an expr anyway.
        assert isinstance(expr, ir.Expr)

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
class CommonSubexpressionElimination(PreserveLocationVisitor, NodeTranslator):
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
    uids: UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: UIDGenerator(prefix="_cs")
    )

    collect_all: bool = dataclasses.field(default=False)

    def visit_FunCall(self, node: ir.FunCall):
        if isinstance(node.fun, ir.SymRef) and node.fun.id in [
            "cartesian_domain",
            "unstructured_domain",
        ]:
            return node

        new_expr, extracted, ignored_children = extract_subexpression(
            node, lambda subexpr, num_occurences: num_occurences > 1, self.uids
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
