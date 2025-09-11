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
import re
from typing import Literal, Optional

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir, ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import fixed_point_transformation, inline_lambdas, inline_lifts
from gt4py.next.iterator.type_system import (
    inference as itir_type_inference,
    type_specifications as it_ts,
)
from gt4py.next.type_system import type_info, type_specifications as ts


def _with_altered_iterator_element_type(
    type_: it_ts.IteratorType, new_el_type: ts.DataType
) -> it_ts.IteratorType:
    return it_ts.IteratorType(
        position_dims=type_.position_dims, defined_dims=type_.defined_dims, element_type=new_el_type
    )


def _with_altered_iterator_position_dims(
    type_: it_ts.IteratorType, new_position_dims: list[common.Dimension] | Literal["unknown"]
) -> it_ts.IteratorType:
    return it_ts.IteratorType(
        position_dims=new_position_dims,
        defined_dims=type_.defined_dims,
        element_type=type_.element_type,
    )


def _is_trivial_make_tuple_call(node: itir.Expr):
    """Return if node is a `make_tuple` call with all elements `SymRef`s, `Literal`s or tuples thereof."""
    if not cpm.is_call_to(node, "make_tuple"):
        return False
    if not all(_is_trivial_or_tuple_thereof_expr(arg) for arg in node.args):
        return False
    return True


def _is_trivial_or_tuple_thereof_expr(node: itir.Node) -> bool:
    """
    Return `true` if the expr is a trivial expression (`SymRef` or `Literal`) or tuple thereof.

    Let forms with trivial body and args as well as `if` calls with trivial branches are also
    considered trivial.

    >>> _is_trivial_or_tuple_thereof_expr(im.make_tuple("a", "b"))
    True
    >>> _is_trivial_or_tuple_thereof_expr(im.tuple_get(1, "a"))
    True
    >>> _is_trivial_or_tuple_thereof_expr(
    ...     im.let("t", im.make_tuple("a", "b"))(im.tuple_get(1, "t"))
    ... )
    True
    """
    if isinstance(node, (itir.SymRef, itir.Literal)):
        return True
    if cpm.is_call_to(node, "make_tuple"):
        return all(_is_trivial_or_tuple_thereof_expr(arg) for arg in node.args)
    if cpm.is_call_to(node, "tuple_get"):
        return _is_trivial_or_tuple_thereof_expr(node.args[1])
    # This will duplicate the condition and increase the size of the tree, but this is probably
    # acceptable.
    if cpm.is_call_to(node, "if_"):
        return all(_is_trivial_or_tuple_thereof_expr(arg) for arg in node.args[1:])
    if cpm.is_let(node):
        return _is_trivial_or_tuple_thereof_expr(node.fun.expr) and all(
            _is_trivial_or_tuple_thereof_expr(arg) for arg in node.args
        )
    return False


def _flattened_as_fieldop_param_el_name(param: str, idx: int) -> str:
    prefix = "__ct_flat_el_"

    # keep the original param name, but skip prefix from previous flattenings
    if param.startswith(prefix):
        parent_idx, suffix = re.split(r"_(?!\d)", param[len(prefix) :], maxsplit=1)
        prefix = f"{prefix}{parent_idx}_"
    else:
        suffix = param

    return f"{prefix}{idx}_{suffix}"


# TODO(tehrengruber): Conceptually the structure of this pass makes sense: Visit depth first,
#  transform each node until no transformations apply anymore, whenever a node is to be transformed
#  go through all available transformation and apply them. However the final result here still
#  reads a little convoluted and is also different to how we write other transformations. We
#  should revisit the pattern here and try to find a more general mechanism.
@dataclasses.dataclass(frozen=True, kw_only=True)
class CollapseTuple(
    fixed_point_transformation.CombinedFixedPointTransform, eve.PreserveLocationVisitor
):
    """
    Simplifies `make_tuple`, `tuple_get` calls.

      - `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
      - `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
    """

    # TODO(tehrengruber): This Flag mechanism is a little low level. What we actually want
    #   is something like a pass manager, where for each pattern we have a corresponding
    #   transformation, etc.
    class Transformation(enum.Flag):
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
        #:  Similar as `PROPAGATE_TO_IF_ON_TUPLES`, but propagates in the opposite direction, i.e.
        #:  into the tree, allowing removal of tuple expressions across `if_` calls without
        #:  increasing the size of the tree. This is particularly important for `if` statements
        #:  in the frontend, where outwards propagation can have devastating effects on the tree
        #:  size, without any gained optimization potential. For example
        #:  ```
        #:   complex_lambda(if cond1
        #:     if cond2
        #:       {...}
        #:     else:
        #:       {...}
        #:   else
        #:     {...})
        #:  ```
        #:  is problematic, since `PROPAGATE_TO_IF_ON_TUPLES` would propagate and hence duplicate
        #:  `complex_lambda` three times, while we only want to get rid of the tuple expressions
        #:  inside of the `if_`s.
        #:  Note that this transformation is not mutually exclusive to `PROPAGATE_TO_IF_ON_TUPLES`.
        PROPAGATE_TO_IF_ON_TUPLES_CPS = enum.auto()
        #: `(if cond then {1, 2} else {3, 4})[0]` -> `if cond then {1, 2}[0] else {3, 4}[0]`
        PROPAGATE_TO_IF_ON_TUPLES = enum.auto()
        #: `let((a, let(b, 1)(a_val)))(a)`-> `let(b, 1)(let(a, a_val)(a))`
        PROPAGATE_NESTED_LET = enum.auto()
        #: `let(a, 1)(a)` -> `1` or `let(a, b)(f(a))` -> `f(a)`
        INLINE_TRIVIAL_LET = enum.auto()
        #: `as_fieldop(λ(t) → ·t[0]+·t[1])({a, b})` -> `as_fieldop(λ(a, b) → ·a+·b)(a, b)`
        FLATTEN_AS_FIELDOP_ARGS = enum.auto()
        #: `let(a, b[1])(a)` -> `b[1]`
        INLINE_TRIVIAL_TUPLE_LET_VAR = enum.auto()

        @classmethod
        def all(self) -> CollapseTuple.Transformation:
            return functools.reduce(operator.or_, self.__members__.values())

    uids: eve_utils.UIDGenerator
    enabled_transformations: Transformation = Transformation.all()  # noqa: RUF009 [function-call-in-dataclass-default-argument]

    REINFER_TYPES = True

    PRESERVED_ANNEX_ATTRS = ("type", "domain")

    @classmethod
    def apply(
        cls,
        node: itir.Node,
        *,
        remove_letified_make_tuple_elements: bool = True,
        offset_provider_type: Optional[common.OffsetProviderType] = None,
        within_stencil: Optional[bool] = None,
        # manually passing enabled transformations is mostly for allowing separate testing of the modes
        enabled_transformations: Optional[Transformation] = None,
        # allow sym references without a symbol declaration, mostly for testing
        allow_undeclared_symbols: bool = False,
        uids: Optional[eve_utils.UIDGenerator] = None,
    ) -> itir.Node:
        """
        Simplifies `make_tuple`, `tuple_get` calls.

        Arguments:
            node: The node to transform.

        Keyword arguments:
            remove_letified_make_tuple_elements: Run `InlineLambdas` as a post-processing step
                to remove left-overs from `LETIFY_MAKE_TUPLE_ELEMENTS` transformation.
                `(λ(_tuple_el_1, _tuple_el_2) → {_tuple_el_1, _tuple_el_2})(1, 2)` -> {1, 2}`
        """
        enabled_transformations = enabled_transformations or cls.enabled_transformations
        offset_provider_type = offset_provider_type or {}
        uids = uids or eve_utils.UIDGenerator()

        if isinstance(node, itir.Program):
            within_stencil = False
        assert within_stencil in [
            True,
            False,
        ], "Parameter 'within_stencil' mandatory if node is not a 'Program'."

        requires_types = False
        if enabled_transformations & (
            cls.Transformation.PROPAGATE_TO_IF_ON_TUPLES_CPS
            | cls.Transformation.FLATTEN_AS_FIELDOP_ARGS
        ):
            requires_types = True

        if requires_types:
            node = itir_type_inference.infer(
                node,
                offset_provider_type=offset_provider_type,
                allow_undeclared_symbols=allow_undeclared_symbols,
            )

        new_node = cls(
            enabled_transformations=enabled_transformations,
            uids=uids,
        ).visit(node, within_stencil=within_stencil)

        # inline to remove left-overs from LETIFY_MAKE_TUPLE_ELEMENTS. this is important
        # as otherwise two equal expressions containing a tuple will not be equal anymore
        # and the CSE pass can not remove them.
        # TODO(tehrengruber): test case for `scan(lambda carry: {1, 2})`
        #  (see solve_nonhydro_stencil_52_like_z_q_tup)
        if remove_letified_make_tuple_elements:
            new_node = inline_lambdas.InlineLambdas.apply(
                new_node, opcount_preserving=True, force_inline_lambda_args=False
            )

        return new_node

    def visit(self, node, **kwargs):
        if cpm.is_call_to(node, "as_fieldop"):
            kwargs = {**kwargs, "within_stencil": True}

        return super().visit(node, **kwargs)

    def transform_collapse_make_tuple_tuple_get(
        self, node: itir.FunCall, **kwargs
    ) -> Optional[itir.Node]:
        if cpm.is_call_to(node, "make_tuple") and all(
            cpm.is_call_to(arg, "tuple_get") for arg in node.args
        ):
            # `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> `t`
            assert len(node.args) > 0 and isinstance(node.args[0], itir.FunCall)
            first_expr = node.args[0].args[1]

            for i, v in enumerate(node.args):
                assert isinstance(v, itir.FunCall)
                assert isinstance(v.args[0], itir.Literal)
                if not (int(v.args[0].value) == i and ir_misc.is_equal(v.args[1], first_expr)):
                    # tuple argument differs, just continue with the rest of the tree
                    return None

            itir_type_inference.reinfer(first_expr)  # type is needed so reinfer on-demand
            assert isinstance(first_expr.type, (ts.TupleType, ts.DeferredType))
            if isinstance(first_expr.type, ts.TupleType) and len(first_expr.type.types) == len(
                node.args
            ):
                return first_expr
        return None

    def transform_collapse_tuple_get_make_tuple(
        self, node: itir.FunCall, **kwargs
    ) -> Optional[itir.Node]:
        if (
            cpm.is_call_to(node, "tuple_get")
            and isinstance(node.args[0], itir.Literal)
            and cpm.is_call_to(node.args[1], "make_tuple")
        ):
            # `tuple_get(i, make_tuple(e_0, e_1, ..., e_i, ..., e_N))` -> `e_i`
            assert not node.args[0].type or type_info.is_integer(node.args[0].type)
            make_tuple_call = node.args[1]
            idx = int(node.args[0].value)
            assert idx < len(make_tuple_call.args), (
                f"Index {idx} is out of bounds for tuple of size {len(make_tuple_call.args)}"
            )
            return node.args[1].args[idx]
        return None

    def transform_propagate_tuple_get(self, node: itir.FunCall, **kwargs) -> Optional[itir.Node]:
        if cpm.is_call_to(node, "tuple_get") and isinstance(node.args[0], itir.Literal):
            # TODO(tehrengruber): extend to general symbols as long as the tail call in the let
            #   does not capture
            # `tuple_get(i, let(...)(make_tuple()))` -> `let(...)(tuple_get(i, make_tuple()))`
            idx, expr = node.args
            assert isinstance(idx, itir.Literal)
            if cpm.is_let(expr):
                return im.call(
                    im.lambda_(*expr.fun.params)(
                        self.fp_transform(im.tuple_get(idx.value, expr.fun.expr), **kwargs)
                    )
                )(*expr.args)
            elif cpm.is_call_to(expr, ("if_", "concat_where")):
                fun = expr.fun
                cond, true_branch, false_branch = expr.args
                return im.call(fun)(
                    cond,
                    self.fp_transform(im.tuple_get(idx.value, true_branch), **kwargs),
                    self.fp_transform(im.tuple_get(idx.value, false_branch), **kwargs),
                )
        return None

    def transform_letify_make_tuple_elements(
        self, node: itir.Node, **kwargs
    ) -> Optional[itir.Node]:
        if cpm.is_call_to(node, "make_tuple"):
            # `make_tuple(expr1, expr1)`
            # -> `let((_tuple_el_1, expr1), (_tuple_el_2, expr2))(make_tuple(_tuple_el_1, _tuple_el_2))`
            bound_vars: dict[itir.Sym, itir.Expr] = {}
            new_args: list[itir.Expr] = []
            for arg in node.args:
                if cpm.is_call_to(node, "make_tuple") and not _is_trivial_make_tuple_call(node):
                    new_arg = im.ref(self.uids.sequential_id(prefix="__ct_el"), arg.type)
                    self._preserve_annex(arg, new_arg)
                    new_args.append(new_arg)
                    bound_vars[im.sym(new_arg.id, arg.type)] = arg
                else:
                    new_args.append(arg)

            if bound_vars:
                return self.fp_transform(
                    im.let(*bound_vars.items())(im.call(node.fun)(*new_args)), **kwargs
                )
        return None

    def transform_inline_trivial_make_tuple(self, node: itir.Node, **kwargs) -> Optional[itir.Node]:
        if cpm.is_let(node):
            # `let(tup, make_tuple(trivial_expr1, trivial_expr2))(foo(tup))`
            #  -> `foo(make_tuple(trivial_expr1, trivial_expr2))`
            eligible_params = [_is_trivial_make_tuple_call(arg) for arg in node.args]
            if any(eligible_params):
                return self.visit(
                    inline_lambdas.inline_lambda(node, eligible_params=eligible_params), **kwargs
                )
        return None

    def transform_propagate_to_if_on_tuples(
        self, node: itir.FunCall, **kwargs
    ) -> Optional[itir.Node]:
        if kwargs["within_stencil"]:
            # TODO(tehrengruber): This significantly increases the size of the tree. Skip transformation
            #  in local-view for now. Revisit.
            return None

        if isinstance(node, itir.FunCall) and not cpm.is_call_to(node, "if_"):
            # TODO(tehrengruber): Only inline if type of branch value is a tuple.
            # Examples:
            # `(if cond then {1, 2} else {3, 4})[0]` -> `if cond then {1, 2}[0] else {3, 4}[0]`
            # `let (b, if cond then {1, 2} else {3, 4})) b[0]`
            #  -> `if cond then let(b, {1, 2})(b[0]) else let(b, {3, 4})(b[0])`
            for i, arg in enumerate(node.args):
                if cpm.is_call_to(arg, "if_"):
                    cond, true_branch, false_branch = arg.args
                    new_true_branch = self.fp_transform(
                        ir_misc.with_altered_arg(node, i, true_branch), **kwargs
                    )
                    new_false_branch = self.fp_transform(
                        ir_misc.with_altered_arg(node, i, false_branch), **kwargs
                    )
                    return im.if_(cond, new_true_branch, new_false_branch)
        return None

    def transform_propagate_to_if_on_tuples_cps(
        self, node: itir.FunCall, **kwargs
    ) -> Optional[itir.Node]:
        # The basic idea of this transformation is to remove tuples across if-stmts by rewriting
        # the expression in continuation passing style, e.g. something like a tuple reordering
        # ```
        # let t = if True then {1, 2} else {3, 4} in
        #   {t[1], t[0]})
        # end
        # ```
        # is rewritten into:
        # ```
        # let cont = λ(el0, el1) → {el1, el0} in
        #  if True then cont(1, 2) else cont(3, 4)
        # end
        # ```
        # Note how the `make_tuple` call argument of the `if` disappears. Since lambda functions
        # are currently inlined (due to limitations of the domain inference) we will only
        # gain something compared `PROPAGATE_TO_IF_ON_TUPLES` if the continuation `cont` is trivial,
        # e.g. a `make_tuple` call like in the example. In that case we can inline the trivial
        # continuation and end up with an only moderately larger tree, e.g.
        # `if True then {2, 1} else {4, 3}`. The examples in the comments below all refer to this
        # tuple reordering example here.

        if not isinstance(node, itir.FunCall) or cpm.is_call_to(node, "if_"):
            return None

        # The first argument that is eligible also transforms all remaining args (They will be
        # part of the continuation which is recursively transformed).
        for i, arg in enumerate(node.args):
            if cpm.is_call_to(arg, "if_"):
                itir_type_inference.reinfer(arg)

                cond, true_branch, false_branch = arg.args  # e.g. `True`, `{1, 2}`, `{3, 4}`
                if not any(
                    isinstance(branch.type, ts.TupleType) for branch in [true_branch, false_branch]
                ):
                    continue
                tuple_type: ts.TupleType = true_branch.type  # type: ignore[assignment]  # type ensured above
                tuple_len = len(tuple_type.types)

                # build and simplify continuation, e.g. λ(el0, el1) → {el1, el0}
                itir_type_inference.reinfer(node)
                assert node.type
                f_type = ts.FunctionType(  # type of continuation in order to keep full type info
                    pos_only_args=tuple_type.types,
                    pos_or_kw_args={},
                    kw_only_args={},
                    returns=node.type,
                )
                f_params = [
                    im.sym(self.uids.sequential_id(prefix="__ct_el_cps"), type_)
                    for type_ in tuple_type.types
                ]
                f_args = [im.ref(param.id, param.type) for param in f_params]
                f_body = ir_misc.with_altered_arg(node, i, im.make_tuple(*f_args))

                # if the function is not trivial the transformation would create a larger tree
                # after inlining so we skip transformation this argument.
                if not _is_trivial_or_tuple_thereof_expr(f_body):
                    continue

                # simplify, e.g., inline trivial make_tuple args
                new_f_body = self.fp_transform(f_body, **kwargs)
                # if the continuation did not simplify there is nothing to gain. Skip
                # transformation of this argument.
                if new_f_body is f_body:
                    continue

                f = im.lambda_(*f_params)(new_f_body)

                # this is the symbol refering to the tuple value inside the two branches of the
                # if, e.g. a symbol refering to `{1, 2}` and `{3, 4}` respectively
                tuple_var = self.uids.sequential_id(prefix="__ct_tuple_cps")
                # this is the symbol refering to our continuation, e.g. `cont` in our example.
                f_var = self.uids.sequential_id(prefix="__ct_cont")
                new_branches = []
                for branch in [true_branch, false_branch]:
                    new_branch = im.let(tuple_var, branch)(
                        im.call(im.ref(f_var, f_type))(  # call to the continuation
                            *(
                                im.tuple_get(i, im.ref(tuple_var, branch.type))
                                for i in range(tuple_len)
                            )
                        )
                    )
                    new_branches.append(self.fp_transform(new_branch, **kwargs))

                # assemble everything together
                new_node = im.let(f_var, f)(im.if_(cond, *new_branches))
                new_node = inline_lambdas.inline_lambda(new_node, eligible_params=[True])
                assert cpm.is_call_to(new_node, "if_")
                new_node = im.if_(
                    cond, *(self.fp_transform(branch, **kwargs) for branch in new_node.args[1:])
                )
                return new_node

        return None

    def transform_propagate_nested_let(self, node: itir.FunCall, **kwargs) -> Optional[itir.Node]:
        if cpm.is_let(node):
            # `let((a, let(b, 1)(a_val)))(a)`-> `let(b, 1)(let(a, a_val)(a))`
            outer_vars: dict[itir.Sym, itir.Expr] = {}
            inner_vars: dict[itir.Sym, itir.Expr] = {}
            original_inner_expr = node.fun.expr
            for arg_sym, arg in zip(node.fun.params, node.args):
                assert arg_sym not in inner_vars
                if cpm.is_let(arg):
                    rename_map: dict[
                        str, ir.SymRef
                    ] = {}  # mapping from symbol with a collision to its new (unique) name
                    for sym, val in zip(arg.fun.params, arg.args):
                        unique_sym = ir_misc.unique_symbol(sym, [s.id for s in outer_vars.keys()])
                        if sym != unique_sym:  # name collision, rename symbol to unique_sym later
                            rename_map[sym.id] = im.ref(unique_sym.id, sym.type)

                        outer_vars[unique_sym] = val

                    new_expr = arg.fun.expr
                    if rename_map:
                        new_expr = inline_lambdas.rename_symbols(new_expr, rename_map)

                    inner_vars[arg_sym] = new_expr
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

    def transform_inline_trivial_let(self, node: itir.FunCall, **kwargs) -> Optional[itir.Node]:
        if cpm.is_let(node):
            if isinstance(node.fun.expr, itir.SymRef):
                # `let(a, 1)(a)` -> `1`
                for arg_sym, arg in zip(node.fun.params, node.args):
                    if isinstance(node.fun.expr, itir.SymRef) and node.fun.expr.id == arg_sym.id:
                        return arg
            if any(
                trivial_args := [isinstance(arg, (itir.SymRef, itir.Literal)) for arg in node.args]
            ):
                return inline_lambdas.inline_lambda(node, eligible_params=trivial_args)

        return None

    def transform_inline_trivial_tuple_let_var(self, node: ir.Node, **kwargs) -> Optional[ir.Node]:
        if cpm.is_let(node):
            if any(trivial_args := [_is_trivial_or_tuple_thereof_expr(arg) for arg in node.args]):
                return inline_lambdas.inline_lambda(node, eligible_params=trivial_args)
        return None

    # TODO(tehrengruber): This is a transformation that should be executed before visiting the children. Then
    #  revisiting the body would not be needed.
    def transform_flatten_as_fieldop_args(
        self, node: itir.FunCall, **kwargs
    ) -> Optional[itir.Node]:
        # `as_fieldop(λ(t) → ·t[0]+·t[1])({a, b})` -> `as_fieldop(λ(a, b) → ·a+·b)(a, b)`
        if not cpm.is_applied_as_fieldop(node):
            return None

        for arg in node.args:
            itir_type_inference.reinfer(arg)

        if not any(isinstance(arg.type, ts.TupleType) for arg in node.args):
            return None

        node = ir_misc.canonicalize_as_fieldop(node)
        stencil, restore_scan = ir_misc.unwrap_scan(
            node.fun.args[0]  # type: ignore[attr-defined] # ensured by cpm.is_applied_as_fieldop
        )

        new_body = stencil.expr
        domain = node.fun.args[1] if len(node.fun.args) > 1 else None  # type: ignore[attr-defined] # ensured by cpm.is_applied_as_fieldop
        remapped_args: dict[
            itir.Sym, itir.Expr
        ] = {}  # contains the arguments that are remapped, e.g. `{a, b}`
        new_params: list[itir.Sym] = []
        new_args: list[itir.Expr] = []
        for param, arg in zip(stencil.params, node.args, strict=True):
            if isinstance(arg.type, ts.TupleType):
                assert isinstance(param.type, it_ts.IteratorType)
                ref_to_remapped_arg = im.ref(
                    f"__ct_flat_remapped_{len(remapped_args)}",
                    arg.type,
                )
                self._preserve_annex(arg, ref_to_remapped_arg)
                remapped_args[im.sym(ref_to_remapped_arg.id, arg.type)] = arg
                new_params_inner, lift_params = [], []
                assert isinstance(param.type.element_type, ts.TupleType)
                for i, type_ in enumerate(param.type.element_type.types):
                    assert isinstance(type_, ts.DataType)
                    new_param = im.sym(
                        _flattened_as_fieldop_param_el_name(param.id, i),
                        _with_altered_iterator_element_type(param.type, type_),
                    )
                    lift_params.append(
                        im.sym(
                            new_param.id,
                            _with_altered_iterator_position_dims(new_param.type, "unknown"),  # type: ignore[arg-type]  # always in IteratorType
                        )
                    )
                    new_params_inner.append(new_param)
                    new_args.append(im.tuple_get(i, ref_to_remapped_arg))

                # an iterator that substitutes the original (tuple) iterator, e.g. `t`. Built
                # from the new parameters which are the elements of `t`.
                param_substitute = im.lift(
                    im.lambda_(*lift_params)(
                        im.make_tuple(*[im.deref(im.ref(p.id, p.type)) for p in lift_params])
                    )
                )(*[im.ref(p.id, p.type) for p in new_params_inner])

                new_body = im.let(param.id, param_substitute)(new_body)
                # note: the lift is trivial so inlining it is not an issue with respect to tree size
                new_body = inline_lambdas.inline_lambda(new_body, force_inline_lift_args=True)

                new_params.extend(new_params_inner)
            else:
                new_params.append(param)
                new_args.append(arg)

        # remove lifts again
        new_body = inline_lifts.InlineLifts(
            flags=inline_lifts.InlineLifts.Flag.INLINE_DEREF_LIFT
            | inline_lifts.InlineLifts.Flag.PROPAGATE_SHIFT
        ).visit(new_body)
        new_body = self.visit(new_body, **kwargs)
        new_stencil = restore_scan(im.lambda_(*new_params)(new_body))

        return im.let(*remapped_args.items())(im.as_fieldop(new_stencil, domain)(*new_args))
