# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.common import OffsetProvider
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import trace_shifts


@dataclasses.dataclass(frozen=True)
class CheckInOutField(PreserveLocationVisitor, NodeTranslator):
    """
    Checks within a SetAt if any fields which are written to are also read with an offset and raises a ValueError in this case.

    Example:
        >>> from gt4py.next.iterator.transforms import infer_domain
        >>> from gt4py.next.type_system import type_specifications as ts
        >>> from gt4py.next.iterator.ir_utils import ir_makers as im
        >>> float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
        >>> IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
        >>> i_field_type = ts.FieldType(dims=[IDim], dtype=float_type)
        >>> offset_provider = {"IOff": IDim}
        >>> cartesian_domain = im.call("cartesian_domain")(
        ...     im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 5)
        ... )
        >>> ir = itir.Program(
        ...     id="test",
        ...     function_definitions=[],
        ...     params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        ...     declarations=[],
        ...     body=[
        ...         itir.SetAt(
        ...             expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
        ...                 im.ref("inout")
        ...             ),
        ...             domain=cartesian_domain,
        ...             target=im.ref("inout"),
        ...         ),
        ...     ],
        ... )
        >>> CheckInOutField.apply(ir, offset_provider=offset_provider)
        Traceback (most recent call last):
        ...
        ValueError: The target inout is also read with an offset.
    """

    @classmethod
    def apply(
        cls,
        program: itir.Program,
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
    ):
        return cls().visit(program, offset_provider=offset_provider)

    def visit_SetAt(self, node: itir.SetAt, **kwargs) -> itir.SetAt:
        offset_provider = kwargs["offset_provider"]

        def extract_subexprs(expr):
            """Return a list of all subexpressions in expr.args, including expr itself."""
            subexprs = [expr]
            if isinstance(expr, itir.FunCall):
                for arg in expr.args:
                    subexprs.extend(extract_subexprs(arg))
            return subexprs

        def visit_nested_make_tuple_tuple_get(expr):
            """Recursively visit make_tuple and tuple_get expr and check all as_fieldop subexpressions."""
            if cpm.is_applied_as_fieldop(expr):
                check_expr(expr.fun, expr.args, offset_provider)
            elif cpm.is_call_to(expr, ("make_tuple", "tuple_get")):
                for arg in expr.args:
                    visit_nested_make_tuple_tuple_get(arg)

        def filter_shifted_args(
            shifts: list[set[tuple[itir.OffsetLiteral, ...]]],
            args: list[itir.Expr],
            offset_provider: OffsetProvider,
        ) -> list[itir.Expr]:
            """
            Filters out trivial shifts (empty or all horizontal/vertical with zero offset)
            and returns filtered shifts and corresponding args.
            """
            filtered = [
                arg
                for shift, arg in zip(shifts, args)
                if shift not in (set(), {()})
                and any(
                    offset_provider[off.value].kind  # type: ignore[index]  # mypy not smart enough
                    not in {common.DimensionKind.HORIZONTAL, common.DimensionKind.VERTICAL}
                    or val.value != 0
                    for off, val in (
                        (pair for pair in shift if len(pair) == 2)  # set case: skip ()
                        if isinstance(shift, set)
                        else zip(shift[0::2], shift[1::2])  # tuple/list case
                    )
                )
            ]
            return filtered if filtered else []

        def check_expr(
            fun: itir.FunCall,
            args: list[itir.Expr],
            offset_provider: OffsetProvider,
        ) -> None:
            shifts = trace_shifts.trace_stencil(fun.args[0], num_args=len(args))

            shifted_args = filter_shifted_args(shifts, args, offset_provider)
            target_subexprs = extract_subexprs(node.target)
            for arg in shifted_args:
                arg_subexprs = extract_subexprs(arg)
                for subexpr in arg_subexprs:
                    if subexpr in target_subexprs:
                        raise ValueError(f"The target {node.target} is also read with an offset.")
                    if not cpm.is_tuple_expr_of(lambda e: isinstance(e, itir.SymRef), arg):
                        raise ValueError(
                            f"Unexpected as_fieldop argument {arg}. Expected `make_tuple`, `tuple_get` or `SymRef`. Please run temporary extraction first."
                        )

        if cpm.is_applied_as_fieldop(node.expr):
            check_expr(node.expr.fun, node.expr.args, offset_provider)
        else:  # Account for nested im.make_tuple and im.tuple_get
            visit_nested_make_tuple_tuple_get(node.expr)
        return node
