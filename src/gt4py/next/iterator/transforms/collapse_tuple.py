# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import functools
import operator
from typing import Optional

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas, inline_lambda
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


def _with_altered_arg(node: ir.FunCall, arg_idx: int, new_arg: ir.Expr):
    """Given a itir.FunCall return a new call with one of its argument replaced."""
    return ir.FunCall(
        fun=node.fun, args=[arg if i != arg_idx else new_arg for i, arg in enumerate(node.args)]
    )


def _is_trivial_make_tuple_call(node: ir.Expr):
    """Return if node is a `make_tuple` call with all elements `SymRef`s, `Literal`s or tuples thereof."""
    if not cpm.is_call_to(node, "make_tuple"):
        return False
    if not all(
        isinstance(arg, (ir.SymRef, ir.Literal)) or _is_trivial_make_tuple_call(arg)
        for arg in node.args
    ):
        return False
    return True


# TODO(tehrengruber): Conceptually the structure of this pass makes sense: Visit depth first,
#  transform each node until no transformations apply anymore, whenever a node is to be transformed
#  go through all available transformation and apply them. However the final result here still
#  reads a little convoluted and is also different to how we write other transformations. We
#  should revisit the pattern here and try to find a more general mechanism.
@dataclasses.dataclass(frozen=True)
class CollapseTuple(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """
    Simplifies `make_tuple`, `tuple_get` calls.

      - `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
      - `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
    """

    # TODO(tehrengruber): This Flag mechanism is a little low level. What we actually want
    #   is something like a pass manager, where for each pattern we have a corresponding
    #   transformation, etc.
    class Flag(enum.Flag):
        #: `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
        COLLAPSE_MAKE_TUPLE_TUPLE_GET = enum.auto()
        #: `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
        COLLAPSE_TUPLE_GET_MAKE_TUPLE = enum.auto()
        #: `tuple_get(i, let(...)(make_tuple()))` -> `let(...)(tuple_get(i, make_tuple()))`
        PROPAGATE_TUPLE_GET = enum.auto()
        #: `{1, 2}` -> `(λ(_tuple_el_1, _tuple_el_2) → {_tuple_el_1, _tuple_el_2})(1, 2)`
        LETIFY_MAKE_TUPLE_ELEMENTS = enum.auto()
        #: `let(tup, {trivial_expr1, trivial_expr2})(foo(tup))`
        #:  -> `foo({trivial_expr1, trivial_expr2})`
        INLINE_TRIVIAL_MAKE_TUPLE = enum.auto()
        #: `(if cond then {1, 2} else {3, 4})[0]` -> `if cond then {1, 2}[0] else {3, 4}[0]`
        PROPAGATE_TO_IF_ON_TUPLES = enum.auto()
        #: `let((a, let(b, 1)(a_val)))(a)`-> `let(b, 1)(let(a, a_val)(a))`
        PROPAGATE_NESTED_LET = enum.auto()
        #: `let(a, 1)(a)` -> `1`
        INLINE_TRIVIAL_LET = enum.auto()

        @classmethod
        def all(self) -> CollapseTuple.Flag:
            return functools.reduce(operator.or_, self.__members__.values())

    ignore_tuple_size: bool
    flags: Flag = Flag.all()  # noqa: RUF009 [function-call-in-dataclass-default-argument]

    PRESERVED_ANNEX_ATTRS = ("type",)

    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    _letify_make_tuple_uids: eve_utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=lambda: eve_utils.UIDGenerator(prefix="_tuple_el")
    )

    @classmethod
    def apply(
        cls,
        node: ir.Node,
        *,
        ignore_tuple_size: bool = False,
        remove_letified_make_tuple_elements: bool = True,
        offset_provider_type: Optional[common.OffsetProviderType] = None,
        within_stencil: Optional[bool] = None,
        # manually passing flags is mostly for allowing separate testing of the modes
        flags: Optional[Flag] = None,
        # allow sym references without a symbol declaration, mostly for testing
        allow_undeclared_symbols: bool = False,
    ) -> ir.Node:
        """
        Simplifies `make_tuple`, `tuple_get` calls.

        Arguments:
            node: The node to transform.

        Keyword arguments:
            ignore_tuple_size: Apply the transformation even if length of the inner tuple is greater
                than the length of the outer tuple.
            remove_letified_make_tuple_elements: Run `InlineLambdas` as a post-processing step
                to remove left-overs from `LETIFY_MAKE_TUPLE_ELEMENTS` transformation.
                `(λ(_tuple_el_1, _tuple_el_2) → {_tuple_el_1, _tuple_el_2})(1, 2)` -> {1, 2}`
        """
        flags = flags or cls.flags
        offset_provider_type = offset_provider_type or {}

        if isinstance(node, (ir.Program, ir.FencilDefinition)):
            within_stencil = False
        assert within_stencil in [
            True,
            False,
        ], "Parameter 'within_stencil' mandatory if node is not a 'Program'."

        if not ignore_tuple_size:
            node = itir_type_inference.infer(
                node,
                offset_provider_type=offset_provider_type,
                allow_undeclared_symbols=allow_undeclared_symbols,
            )

        new_node = cls(
            ignore_tuple_size=ignore_tuple_size,
            flags=flags,
        ).visit(node, within_stencil=within_stencil)

        # inline to remove left-overs from LETIFY_MAKE_TUPLE_ELEMENTS. this is important
        # as otherwise two equal expressions containing a tuple will not be equal anymore
        # and the CSE pass can not remove them.
        # TODO(tehrengruber): test case for `scan(lambda carry: {1, 2})`
        #  (see solve_nonhydro_stencil_52_like_z_q_tup)
        if remove_letified_make_tuple_elements:
            new_node = InlineLambdas.apply(
                new_node, opcount_preserving=True, force_inline_lambda_args=False
            )

        return new_node

    def visit_FunCall(self, node: ir.FunCall, **kwargs) -> ir.Node:
        if cpm.is_call_to(node, "as_fieldop"):
            kwargs = {**kwargs, "within_stencil": True}

        node = self.generic_visit(node, **kwargs)
        return self.fp_transform(node, **kwargs)

    def fp_transform(self, node: ir.Node, **kwargs) -> ir.Node:
        while True:
            new_node = self.transform(node, **kwargs)
            if new_node is None:
                break
            assert new_node != node
            node = new_node
        return node

    def transform(self, node: ir.Node, **kwargs) -> Optional[ir.Node]:
        if not isinstance(node, ir.FunCall):
            return None

        for transformation in self.Flag:
            if self.flags & transformation:
                assert isinstance(transformation.name, str)
                method = getattr(self, f"transform_{transformation.name.lower()}")
                result = method(node, **kwargs)
                if result is not None:
                    return result
        return None

    def transform_collapse_make_tuple_tuple_get(
        self, node: ir.FunCall, **kwargs
    ) -> Optional[ir.Node]:
        if node.fun == ir.SymRef(id="make_tuple") and all(
            isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="tuple_get")
            for arg in node.args
        ):
            # `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
            assert isinstance(node.args[0], ir.FunCall)
            first_expr = node.args[0].args[1]

            for i, v in enumerate(node.args):
                assert isinstance(v, ir.FunCall)
                assert isinstance(v.args[0], ir.Literal)
                if not (int(v.args[0].value) == i and ir_misc.is_equal(v.args[1], first_expr)):
                    # tuple argument differs, just continue with the rest of the tree
                    return None

            assert self.ignore_tuple_size or isinstance(
                first_expr.type, (ts.TupleType, ts.DeferredType)
            )
            if self.ignore_tuple_size or (
                isinstance(first_expr.type, ts.TupleType)
                and len(first_expr.type.types) == len(node.args)
            ):
                return first_expr
        return None

    def transform_collapse_tuple_get_make_tuple(
        self, node: ir.FunCall, **kwargs
    ) -> Optional[ir.Node]:
        if (
            node.fun == ir.SymRef(id="tuple_get")
            and isinstance(node.args[1], ir.FunCall)
            and node.args[1].fun == ir.SymRef(id="make_tuple")
            and isinstance(node.args[0], ir.Literal)
        ):
            # `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
            assert type_info.is_integer(node.args[0].type)
            make_tuple_call = node.args[1]
            idx = int(node.args[0].value)
            assert idx < len(
                make_tuple_call.args
            ), f"Index {idx} is out of bounds for tuple of size {len(make_tuple_call.args)}"
            return node.args[1].args[idx]
        return None

    def transform_propagate_tuple_get(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if node.fun == ir.SymRef(id="tuple_get") and isinstance(node.args[0], ir.Literal):
            # TODO(tehrengruber): extend to general symbols as long as the tail call in the let
            #   does not capture
            # `tuple_get(i, let(...)(make_tuple()))` -> `let(...)(tuple_get(i, make_tuple()))`
            if cpm.is_let(node.args[1]):
                idx, let_expr = node.args
                return im.call(
                    im.lambda_(*let_expr.fun.params)(  # type: ignore[attr-defined]  # ensured by is_let
                        self.fp_transform(im.tuple_get(idx.value, let_expr.fun.expr), **kwargs)  # type: ignore[attr-defined]  # ensured by is_let
                    )
                )(
                    *let_expr.args  # type: ignore[attr-defined]  # ensured by is_let
                )
            elif cpm.is_call_to(node.args[1], "if_"):
                idx = node.args[0]
                cond, true_branch, false_branch = node.args[1].args
                return im.if_(
                    cond,
                    self.fp_transform(im.tuple_get(idx.value, true_branch), **kwargs),
                    self.fp_transform(im.tuple_get(idx.value, false_branch), **kwargs),
                )
        return None

    def transform_letify_make_tuple_elements(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if node.fun == ir.SymRef(id="make_tuple"):
            # `make_tuple(expr1, expr1)`
            # -> `let((_tuple_el_1, expr1), (_tuple_el_2, expr2))(make_tuple(_tuple_el_1, _tuple_el_2))`
            bound_vars: dict[str, ir.Expr] = {}
            new_args: list[ir.Expr] = []
            for arg in node.args:
                if cpm.is_call_to(node, "make_tuple") and not _is_trivial_make_tuple_call(node):
                    el_name = self._letify_make_tuple_uids.sequential_id()
                    new_args.append(im.ref(el_name))
                    bound_vars[el_name] = arg
                else:
                    new_args.append(arg)

            if bound_vars:
                return self.fp_transform(
                    im.let(*bound_vars.items())(im.call(node.fun)(*new_args)), **kwargs
                )
        return None

    def transform_inline_trivial_make_tuple(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if cpm.is_let(node):
            # `let(tup, make_tuple(trivial_expr1, trivial_expr2))(foo(tup))`
            #  -> `foo(make_tuple(trivial_expr1, trivial_expr2))`
            eligible_params = [_is_trivial_make_tuple_call(arg) for arg in node.args]
            if any(eligible_params):
                return self.visit(inline_lambda(node, eligible_params=eligible_params), **kwargs)
        return None

    def transform_propagate_to_if_on_tuples(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if kwargs["within_stencil"]:
            # TODO(tehrengruber): This significantly increases the size of the tree. Skip transformation
            #  in local-view for now. Revisit.
            return None

        if not cpm.is_call_to(node, "if_"):
            # TODO(tehrengruber): Only inline if type of branch value is a tuple.
            # Examples:
            # `(if cond then {1, 2} else {3, 4})[0]` -> `if cond then {1, 2}[0] else {3, 4}[0]`
            # `let (b, if cond then {1, 2} else {3, 4})) b[0]`
            #  -> `if cond then let(b, {1, 2})(b[0]) else let(b, {3, 4})(b[0])`
            for i, arg in enumerate(node.args):
                if cpm.is_call_to(arg, "if_"):
                    cond, true_branch, false_branch = arg.args
                    new_true_branch = self.fp_transform(
                        _with_altered_arg(node, i, true_branch), **kwargs
                    )
                    new_false_branch = self.fp_transform(
                        _with_altered_arg(node, i, false_branch), **kwargs
                    )
                    return im.if_(cond, new_true_branch, new_false_branch)
        return None

    def transform_propagate_nested_let(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if cpm.is_let(node):
            # `let((a, let(b, 1)(a_val)))(a)`-> `let(b, 1)(let(a, a_val)(a))`
            outer_vars = {}
            inner_vars = {}
            original_inner_expr = node.fun.expr  # type: ignore[attr-defined]  # ensured by is_let
            for arg_sym, arg in zip(node.fun.params, node.args):  # type: ignore[attr-defined]  # ensured by is_let
                assert arg_sym not in inner_vars  # TODO(tehrengruber): fix collisions
                if cpm.is_let(arg):
                    for sym, val in zip(arg.fun.params, arg.args):  # type: ignore[attr-defined]  # ensured by is_let
                        assert sym not in outer_vars  # TODO(tehrengruber): fix collisions
                        outer_vars[sym] = val
                    inner_vars[arg_sym] = arg.fun.expr  # type: ignore[attr-defined]  # ensured by is_let
                else:
                    inner_vars[arg_sym] = arg
            if outer_vars:
                return self.fp_transform(
                    im.let(*outer_vars.items())(
                        self.fp_transform(
                            im.let(*inner_vars.items())(original_inner_expr), **kwargs
                        )
                    ),
                    **kwargs,
                )
        return None

    def transform_inline_trivial_let(self, node: ir.FunCall, **kwargs) -> Optional[ir.Node]:
        if cpm.is_let(node) and isinstance(node.fun.expr, ir.SymRef):  # type: ignore[attr-defined]  # ensured by is_let
            # `let(a, 1)(a)` -> `1`
            for arg_sym, arg in zip(node.fun.params, node.args):  # type: ignore[attr-defined]  # ensured by is_let
                if isinstance(node.fun.expr, ir.SymRef) and node.fun.expr.id == arg_sym.id:  # type: ignore[attr-defined]  # ensured by is_let
                    return arg
        return None
