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
import pytest


"""Contains definition of test-exclusion matrices, see ADR 15."""

# Skip definitions
XFAIL = pytest.xfail
SKIP = pytest.skip

# Processor ids as returned by next_tests.get_processor_id()
DACE = "dace_iterator.run_dace_iterator"
GTFN_CPU = "otf_compile_executor.run_gtfn"
GTFN_CPU_IMPERATIVE = "otf_compile_executor.run_gtfn_imperative"
GTFN_CPU_WITH_TEMPORARIES = "otf_compile_executor.run_gtfn_with_temporaries"
GTFN_FORMAT_SOURCECODE = "gtfn.format_sourcecode"

# Test markers
REQUIRES_ATLAS = "requires_atlas"
USES_APPLIED_SHIFTS = "uses_applied_shifts"
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
USES_SPARSE_FIELDS = "uses_sparse_fields"
USES_REDUCTION_WITH_ONLY_SPARSE_FIELDS = "uses_reduction_with_only_sparse_fields"
USES_STRIDED_NEIGHBOR_OFFSET = "uses_strided_neighbor_offset"
USES_TUPLE_ARGS = "uses_tuple_args"
USES_TUPLE_RETURNS = "uses_tuple_returns"
USES_ZERO_DIMENSIONAL_FIELDS = "uses_zero_dimensional_fields"

# Skip messages (available format keys: 'marker', 'backend')
UNSUPPORTED_MESSAGE = "'{marker}' tests not supported by '{backend}' backend"
BINDINGS_UNSUPPORTED_MESSAGE = "'{marker}' not supported by '{backend}' bindings"
REDUCTION_WITH_ONLY_SPARSE_FIELDS_MESSAGE = (
    "We cannot unroll a reduction on a sparse field only (not clear if it is legal ITIR)"
)
# Common list of feature markers to skip
GTFN_SKIP_TEST_LIST = [
    (REQUIRES_ATLAS, XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
    (USES_APPLIED_SHIFTS, XFAIL, UNSUPPORTED_MESSAGE),
    (USES_IF_STMTS, XFAIL, UNSUPPORTED_MESSAGE),
    (USES_NEGATIVE_MODULO, XFAIL, UNSUPPORTED_MESSAGE),
    (USES_REDUCTION_WITH_ONLY_SPARSE_FIELDS, XFAIL, REDUCTION_WITH_ONLY_SPARSE_FIELDS_MESSAGE),
    (USES_SCAN_IN_FIELD_OPERATOR, XFAIL, UNSUPPORTED_MESSAGE),
    (USES_STRIDED_NEIGHBOR_OFFSET, XFAIL, BINDINGS_UNSUPPORTED_MESSAGE),
]

#: Skip matrix, contains for each backend processor a list of tuples with following fields:
#: (<test_marker>, <skip_definition, <skip_message>)
BACKEND_SKIP_TEST_MATRIX = {
    DACE: GTFN_SKIP_TEST_LIST
    + [
        (USES_CAN_DEREF, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_CONSTANT_FIELDS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_DYNAMIC_OFFSETS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_INDEX_FIELDS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_LIFT_EXPRESSIONS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ORIGIN, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_REDUCTION_OVER_LIFT_EXPRESSIONS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_SPARSE_FIELDS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_TUPLE_ARGS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_TUPLE_RETURNS, XFAIL, UNSUPPORTED_MESSAGE),
        (USES_ZERO_DIMENSIONAL_FIELDS, XFAIL, UNSUPPORTED_MESSAGE),
    ],
    GTFN_CPU: GTFN_SKIP_TEST_LIST,
    GTFN_CPU_IMPERATIVE: GTFN_SKIP_TEST_LIST,
    GTFN_CPU_WITH_TEMPORARIES: GTFN_SKIP_TEST_LIST
    + [
        (USES_DYNAMIC_OFFSETS, XFAIL, UNSUPPORTED_MESSAGE),
    ],
    GTFN_FORMAT_SOURCECODE: [
        (USES_REDUCTION_WITH_ONLY_SPARSE_FIELDS, XFAIL, REDUCTION_WITH_ONLY_SPARSE_FIELDS_MESSAGE),
    ],
}
