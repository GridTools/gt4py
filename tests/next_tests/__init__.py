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
import enum

import pytest


# Skip definitions
class SkipMark(enum.Enum):
    XFAIL = pytest.xfail
    SKIP = pytest.skip


# Skip messages (available format keys: 'marker', 'backend')
UNSUPPORTED_MESSAGE = "'{marker}' tests not supported by '{backend}' backend"
BINDINGS_UNSUPPORTED_MESSAGE = "'{marker}' not supported by '{backend}' bindings"

# Processors
DACE = "dace_iterator.run_dace_iterator"
GTFN_CPU = "otf_compile_executor.run_gtfn"
GTFN_IMPERATIVE = "otf_compile_executor.run_gtfn_imperative"
GTFN_CPU_WITH_TEMP = "otf_compile_executor.run_gtfn_with_temporaries"

# Test markers
USES_APPLIED_SHIFTS = "uses_applied_shifts"
USES_ATLAS_TABLES = "uses_atlas_tables"
USES_CAN_DEREF = "uses_can_deref"
USES_CONSTANT_FIELDS = "uses_constant_fields"
USES_DYNAMIC_OFFSETS = "uses_dynamic_offsets"
USES_IF_STMTS = "uses_if_stmts"
USES_INDEX_FIELDS = "uses_index_fields"
USES_LIFT_EXPRESSIONS = "uses_lift_expressions"
USES_NEGATIVE_MODULO = "uses_negative_modulo"
USES_ORIGIN = "uses_origin"
USES_REDUCTION_OVER_LIFT_EXPRESSIONS = "uses_reduction_over_lift_expressions"
USES_SCAN_IN_FIELD_OPERATOR = "uses_scan_in_field_operator"
USES_STRIDED_NEIGHBOR_OFFSET = "uses_strided_neighbor_offset"
USES_TUPLE_ARGS = "uses_tuple_args"
USES_TUPLE_RETURNS = "uses_tuple_returns"
USES_ZERO_DIMENSIONAL_FIELDS = "uses_zero_dimensional_fields"

# Skip matrix
BACKEND_SKIP_TEST_MATRIX = {
    DACE: [
        (USES_APPLIED_SHIFTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ATLAS_TABLES, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_CAN_DEREF, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_CONSTANT_FIELDS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_DYNAMIC_OFFSETS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_IF_STMTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_INDEX_FIELDS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_LIFT_EXPRESSIONS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_NEGATIVE_MODULO, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ORIGIN, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_REDUCTION_OVER_LIFT_EXPRESSIONS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_SCAN_IN_FIELD_OPERATOR, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_STRIDED_NEIGHBOR_OFFSET, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_TUPLE_ARGS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_TUPLE_RETURNS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ZERO_DIMENSIONAL_FIELDS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
    ],
    GTFN_CPU: [
        (USES_APPLIED_SHIFTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ATLAS_TABLES, SkipMark.XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
        (USES_IF_STMTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_STRIDED_NEIGHBOR_OFFSET, SkipMark.XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
    ],
    GTFN_IMPERATIVE: [
        (USES_APPLIED_SHIFTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ATLAS_TABLES, SkipMark.XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
        (USES_IF_STMTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_STRIDED_NEIGHBOR_OFFSET, SkipMark.XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
    ],
    GTFN_CPU_WITH_TEMP: [
        (USES_APPLIED_SHIFTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ATLAS_TABLES, SkipMark.XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
        (USES_DYNAMIC_OFFSETS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_IF_STMTS, SkipMark.XFAIL, UNSUPPORTED_MESSAGE),
        (USES_STRIDED_NEIGHBOR_OFFSET, SkipMark.XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
    ],
}


def get_processor_id(processor):
    if hasattr(processor, "__module__") and hasattr(processor, "__name__"):
        module_path = processor.__module__.split(".")[-1]
        name = processor.__name__
        return f"{module_path}.{name}"
    return repr(processor)
