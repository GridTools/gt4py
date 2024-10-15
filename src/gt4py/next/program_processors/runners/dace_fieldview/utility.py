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

from gt4py import eve
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.type_system import type_specifications as ts


def get_map_variable(dim: gtx_common.Dimension) -> str:
    """
    Format map variable name based on the naming convention for application-specific SDFG transformations.
    """
    suffix = "dim" if dim.kind == gtx_common.DimensionKind.LOCAL else ""
    # TODO(edopao): raise exception if dim.value is empty
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
    Make the IR compliant with the requirements of lowering to SDFG.

    Applies canonicalization of as_fieldop expressions as well as some temporary workarounds.
    This allows to lower the IR to SDFG for some special cases.
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

            node.args = self.visit(node.args)
            node.fun = self.visit(node.fun)
            return node

    return PatchGTIR().visit(ir)
