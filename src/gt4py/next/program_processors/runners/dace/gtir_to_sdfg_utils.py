# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Dict, Optional, TypeVar

import dace

from gt4py import eve
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace import gtir_python_codegen, gtir_to_sdfg_types
from gt4py.next.type_system import type_specifications as ts


def debug_info(
    node: gtir.Node, *, default: Optional[dace.dtypes.DebugInfo] = None
) -> Optional[dace.dtypes.DebugInfo]:
    """Include the GT4Py node location as debug information in the corresponding SDFG nodes."""
    location = node.location
    if location:
        return dace.dtypes.DebugInfo(
            start_line=location.line,
            start_column=location.column if location.column else 0,
            end_line=location.end_line if location.end_line else -1,
            end_column=location.end_column if location.end_column else 0,
            filename=location.filename,
        )
    return default


def get_arg_symbol_mapping(
    dataname: str, arg: gtir_to_sdfg_types.FieldopResult, sdfg: dace.SDFG
) -> dict[str, dace.symbolic.SymExpr]:
    """
    Helper method to build the mapping from inner to outer SDFG of all symbols
    used for storage of a field or a tuple of fields.

    Args:
        dataname: The storage name inside the nested SDFG.
        arg: The argument field in the parent SDFG.
        sdfg: The parent SDFG where the argument field lives.

    Returns:
        A mapping from inner symbol names to values or symbolic definitions
        in the parent SDFG.
    """
    if arg is None:
        return {}
    if isinstance(arg, gtir_to_sdfg_types.FieldopData):
        return arg.get_symbol_mapping(dataname, sdfg)

    symbol_mapping: dict[str, dace.symbolic.SymExpr] = {}
    for i, elem in enumerate(arg):
        dataname_elem = f"{dataname}_{i}"
        symbol_mapping |= get_arg_symbol_mapping(dataname_elem, elem, sdfg)

    return symbol_mapping


def get_map_variable(dim: gtx_common.Dimension) -> str:
    """
    Format map variable name based on the naming convention for application-specific SDFG transformations.
    """
    suffix = "dim" if dim.kind == gtx_common.DimensionKind.LOCAL else ""
    return f"i_{dim.value}_gtx_{dim.kind}{suffix}"


def make_symbol_tree(tuple_name: str, tuple_type: ts.TupleType) -> tuple[gtir.Sym, ...]:
    """
    Creates a tree representation of the symbols corresponding to the tuple fields.
    The constructed tree preserves the nested nature of the tuple type, if any.

    Examples
    --------
    >>> sty = ts.ScalarType(kind=ts.ScalarKind.INT32)
    >>> fty = ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    >>> t = ts.TupleType(types=[sty, ts.TupleType(types=[fty, sty])])
    >>> assert make_symbol_tree("a", t) == (
    ...     im.sym("a_0", sty),
    ...     (im.sym("a_1_0", fty), im.sym("a_1_1", sty)),
    ... )
    """
    assert all(isinstance(t, ts.DataType) for t in tuple_type.types)
    fields = [(f"{tuple_name}_{i}", field_type) for i, field_type in enumerate(tuple_type.types)]
    return tuple(
        make_symbol_tree(field_name, field_type)  # type: ignore[misc]
        if isinstance(field_type, ts.TupleType)
        else im.sym(field_name, field_type)
        for field_name, field_type in fields
    )


def flatten_tuple_fields(tuple_name: str, tuple_type: ts.TupleType) -> list[gtir.Sym]:
    """
    Creates a list of symbols, annotated with the data type, for all elements of the given tuple.

    Examples
    --------
    >>> sty = ts.ScalarType(kind=ts.ScalarKind.INT32)
    >>> fty = ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32))
    >>> t = ts.TupleType(types=[sty, ts.TupleType(types=[fty, sty])])
    >>> assert flatten_tuple_fields("a", t) == [
    ...     im.sym("a_0", sty),
    ...     im.sym("a_1_0", fty),
    ...     im.sym("a_1_1", sty),
    ... ]
    """
    symbol_tree = make_symbol_tree(tuple_name, tuple_type)
    return list(gtx_utils.flatten_nested_tuple(symbol_tree))


def replace_invalid_symbols(ir: gtir.Program) -> gtir.Program:
    """
    Ensure that all symbols used in the program IR are valid strings (e.g. no unicode-strings).

    If any invalid symbol present, this function returns a copy of the input IR where
    the invalid symbols have been replaced with new names. If all symbols are valid,
    the input IR is returned without copying it.
    """

    class ReplaceSymbols(eve.PreserveLocationVisitor, eve.NodeTranslator):
        PRESERVED_ANNEX_ATTRS = ("domain",)

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

    ir_sym_ids = {str(sym.id) for sym in eve.walk_values(ir).if_isinstance(gtir.Sym).to_set()}
    ir_ssa_uuid = eve.utils.UIDGenerator(prefix="gtir_tmp")

    invalid_symbols_mapping = {
        sym_id: ir_ssa_uuid.sequential_id()
        for sym_id in ir_sym_ids
        if not dace.dtypes.validate_name(sym_id)
    }
    if len(invalid_symbols_mapping) == 0:
        return ir

    # assert that the new symbol names are not used in the IR
    assert ir_sym_ids.isdisjoint(invalid_symbols_mapping.values())
    return ReplaceSymbols().visit(ir, symtable=invalid_symbols_mapping)


def get_symbolic(ir: gtir.Expr) -> dace.symbolic.SymbolicType:
    """
    Specialized visit method for symbolic expressions.

    Returns:
        A dace symbolic expression of the given GTIR.
    """
    python_source = gtir_python_codegen.get_source(ir)
    return dace.symbolic.pystr_to_symbolic(python_source)
