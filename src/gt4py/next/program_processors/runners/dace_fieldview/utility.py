# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
from typing import Dict, TypeVar

import dace

from gt4py import eve
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.type_system import type_specifications as ts


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


def replace_invalid_symbols(sdfg: dace.SDFG, ir: gtir.Program) -> gtir.Program:
    """
    Ensure that all symbols used in the program IR are valid strings (e.g. no unicode-strings).

    If any invalid symbol present, this funtion returns a copy of the input IR where
    the invalid symbols have been replaced with new names. If all symbols are valid,
    the input IR is returned without copying it.
    """

    class ReplaceSymbols(eve.PreserveLocationVisitor, eve.NodeTranslator):
        T = TypeVar("T", gtir.Sym, gtir.SymRef)

        def _replace_sym(self, node: T, symtable: Dict[str, str]) -> T:
            sym = str(node.id)
            return type(node)(id=symtable.get(sym, sym), type=node.type)

        def visit_Sym(self, node: gtir.Sym, *, symtable: Dict[str, str]) -> gtir.Sym:
            return self._replace_sym(node, symtable)

        def visit_SymRef(self, node: gtir.SymRef, *, symtable: Dict[str, str]) -> gtir.SymRef:
            return self._replace_sym(node, symtable)

    # program arguments are checked separetely, because they cannot be replaced
    if not all(dace.dtypes.validate_name(str(sym.id)) for sym in ir.params):
        raise ValueError("Invalid symbol in program parameters.")

    invalid_symbols_mapping = {
        sym_id: sdfg.temp_data_name()
        for sym in eve.walk_values(ir).if_isinstance(gtir.Sym).to_set()
        if not dace.dtypes.validate_name(sym_id := str(sym.id))
    }
    if len(invalid_symbols_mapping) != 0:
        return ReplaceSymbols().visit(ir, symtable=invalid_symbols_mapping)
    else:
        return ir
