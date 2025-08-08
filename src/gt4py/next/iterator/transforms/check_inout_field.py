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

        def check_expr(fun, args, offset_provider):
            shifts = trace_shifts.trace_stencil(fun, num_args=len(args))
            target_subexprs = extract_subexprs(node.target)
            for arg, shift in zip(args, shifts):
                arg_subexprs = extract_subexprs(arg)
                for subexpr in arg_subexprs:
                    if subexpr in target_subexprs:
                        if shift not in (set(), {()}):
                            # This condition is just to filter out the trivial offsets in the horizontal and vertical. #  TODO: remove and add preprocessing of IOff(0) instead
                            if any(
                                offset_provider[off.value].kind
                                not in {
                                    common.DimensionKind.HORIZONTAL,
                                    common.DimensionKind.VERTICAL,
                                }
                                or val.value != 0
                                for off, val in shift
                            ):
                                raise ValueError(
                                    f"The target {node.target} is also read with an offset."
                                )
                    if not cpm.is_tuple_expr_of(lambda e: isinstance(e, itir.SymRef), arg):
                        raise ValueError(
                            f"Unexpected as_fieldop argument {arg}. Expected `make_tuple`, `tuple_get` or `SymRef`. Please run temporary extraction first."
                        )

        if cpm.is_applied_as_fieldop(node.expr):
            check_expr(node.expr.fun, node.expr.args, offset_provider)
        else:  # Account for nested im.make_tuple and im.tuple_get
            visit_nested_make_tuple_tuple_get(node.expr)
        return node
