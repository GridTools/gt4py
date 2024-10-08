# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from typing import Any

import dace

from gt4py import eve
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.program_processors.runners.dace_fieldview import gtir_python_codegen
from gt4py.next.type_system import type_specifications as ts


def get_domain(
    node: gtir.Expr,
) -> list[tuple[gtx_common.Dimension, dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Specialized visit method for domain expressions.

    Returns for each domain dimension the corresponding range.

    TODO: Domain expressions will be recurrent in the GTIR program. An interesting idea
          would be to cache the results of lowering here (e.g. using `functools.lru_cache`)
    """
    assert cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))

    domain = []
    for named_range in node.args:
        assert cpm.is_call_to(named_range, "named_range")
        assert len(named_range.args) == 3
        axis = named_range.args[0]
        assert isinstance(axis, gtir.AxisLiteral)
        bounds = [
            dace.symbolic.pystr_to_symbolic(gtir_python_codegen.get_source(arg))
            for arg in named_range.args[1:3]
        ]
        dim = gtx_common.Dimension(axis.value, axis.kind)
        domain.append((dim, bounds[0], bounds[1]))

    return domain


def get_domain_ranges(
    node: gtir.Expr,
) -> dict[gtx_common.Dimension, tuple[dace.symbolic.SymbolicType, dace.symbolic.SymbolicType]]:
    """
    Returns domain represented in dictionary form.
    """
    domain = get_domain(node)

    return {dim: (lb, ub) for dim, lb, ub in domain}


def get_map_variable(dim: gtx_common.Dimension) -> str:
    """
    Format map variable name based on the naming convention for application-specific SDFG transformations.
    """
    suffix = "dim" if dim.kind == gtx_common.DimensionKind.LOCAL else ""
    return f"i_{dim.value}_gtx_{dim.kind}{suffix}"


def get_tuple_fields(
    tuple_name: str, tuple_type: ts.TupleType, flatten: bool = False
) -> list[tuple[str, ts.DataType]]:
    """
    Creates a list of names with the corresponding data type for all elements of the given tuple.

    Examples
    --------
    >>> sty = ts.ScalarType(kind=ts.ScalarKind.INT32)
    >>> fty = ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    >>> t = ts.TupleType(types=[sty, ts.TupleType(types=[fty, sty])])
    >>> assert get_tuple_fields("a", t) == [("a_0", sty), ("a_1", ts.TupleType(types=[fty, sty]))]
    >>> assert get_tuple_fields("a", t, flatten=True) == [
    ...     ("a_0", sty),
    ...     ("a_1_0", fty),
    ...     ("a_1_1", sty),
    ... ]
    """
    fields = [(f"{tuple_name}_{i}", field_type) for i, field_type in enumerate(tuple_type.types)]
    if flatten:
        expanded_fields = [
            get_tuple_fields(field_name, field_type)
            if isinstance(field_type, ts.TupleType)
            else [(field_name, field_type)]
            for field_name, field_type in fields
        ]
        return list(itertools.chain(*expanded_fields))
    else:
        return fields


def get_tuple_type(data: tuple[Any, ...]) -> ts.TupleType:
    """
    Compute the `ts.TupleType` corresponding to the structure of a tuple of data nodes.
    """
    return ts.TupleType(
        types=[get_tuple_type(d) if isinstance(d, tuple) else d.data_type for d in data]
    )


def patch_gtir(ir: gtir.Program) -> gtir.Program:
    """
    Make the IR comply with the requirements of lowering to SDFG.

    Applies canonicalization of as_fieldop expressions as well as some temporary workarounds.
    """

    class PatchGTIR(eve.PreserveLocationVisitor, eve.NodeTranslator):
        def visit_FunCall(self, node: gtir.FunCall) -> gtir.Node:
            if cpm.is_applied_as_fieldop(node):
                assert isinstance(node.fun, gtir.FunCall)
                assert isinstance(node.type, ts.FieldType)

                # Handle the case of fieldop without domain. This case should never happen, but domain
                # inference currently produces this kind of nodes for unreferenced tuple fields.
                # TODO(tehrengruber): remove this workaround once domain ineference supports this case
                if len(node.fun.args) == 1:
                    return gtir.Literal(value="0", type=node.type.dtype)

                assert len(node.fun.args) == 2
                stencil = node.fun.args[0]

                # Canonicalize as_fieldop: always expect a lambda expression.
                # Here we replace the call to deref with a lambda expression without arguments.
                if cpm.is_ref_to(stencil, "deref"):
                    node.fun.args[0] = gtir.Lambda(
                        expr=gtir.FunCall(fun=stencil, args=node.args), params=[]
                    )
                    node.args = []

            node.args = [self.visit(arg) for arg in node.args]
            node.fun = self.visit(node.fun)
            return node

    return PatchGTIR().visit(ir)
