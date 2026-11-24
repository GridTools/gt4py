# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the program linters, which run when a program is defined, for every backend."""

import pytest

import gt4py.next as gtx
from gt4py.next import errors, neighbor_sum
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.program_processors.runners import roundtrip


Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
E2EDim = gtx.Dimension("E2E", kind=gtx.DimensionKind.LOCAL)
Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))
E2E = gtx.FieldOffset("E2E", source=Edge, target=(Edge, E2EDim))

CellKField = gtx.Field[gtx.Dims[Cell, KDim], gtx.float64]
EdgeField = gtx.Field[gtx.Dims[Edge], gtx.float64]
KIntField = gtx.Field[gtx.Dims[Cell, KDim], gtx.int32]

# Defining a program does not compile it, so this runs without a C++ compiler.
BACKENDS = [None, gtx.gtfn_cpu, roundtrip.default]


@gtx.field_operator
def copy_from_below(x: CellKField) -> CellKField:
    return x(Koff[1])


@gtx.field_operator
def copy_from_above(x: CellKField) -> CellKField:
    return x(Koff[-1])


@gtx.field_operator
def scaled_from_below(x: CellKField) -> CellKField:
    return x(Koff[1]) * 2.0


@gtx.field_operator
def sum_with_other(x: CellKField, s: CellKField) -> CellKField:
    return x(Koff[1]) + s


@gtx.field_operator
def scaled_copy_from_below(x: CellKField) -> CellKField:
    return copy_from_below(x) * 2.0


@gtx.field_operator
def copy_and_scale(x: CellKField, y: CellKField) -> tuple[CellKField, CellKField]:
    return x(Koff[1]), y * 3.0


@gtx.field_operator
def shifted_and_incremented(x: CellKField) -> tuple[CellKField, CellKField]:
    return x(Koff[1]), x + 1.0


@gtx.field_operator
def shifted_unless_flag(x: CellKField, flag: bool) -> CellKField:
    if flag:
        return x
    return x(Koff[1])


@gtx.field_operator
def incremented_unless_flag(x: CellKField, flag: bool) -> CellKField:
    if flag:
        return x
    return x + 1.0


@gtx.field_operator
def neighbor_sum_of(x: EdgeField) -> EdgeField:
    return neighbor_sum(x(E2E), axis=E2EDim)


@gtx.field_operator
def dynamically_shifted(x: CellKField, offsets: KIntField) -> CellKField:
    return x(as_offset(Koff, offsets))


@gtx.field_operator
def incremented(x: CellKField) -> CellKField:
    return x + 1.0


@gtx.field_operator
def passthrough_and_use(x: CellKField, y: CellKField) -> tuple[CellKField, CellKField]:
    return y * (x(Koff[-1]) - x), x


@gtx.field_operator
def forward_unless_flag(x: CellKField, y: CellKField, flag: bool) -> tuple[CellKField, CellKField]:
    # `x` is either recomputed from `y`, or forwarded untouched
    x = y * 2.0 if flag else x
    return x, x(Koff[1])


@gtx.field_operator
def zero_shift(x: CellKField) -> CellKField:
    return x(Koff[0])


@gtx.field_operator
def shift_and_shift_back(x: CellKField) -> CellKField:
    return x(Koff[1])(Koff[-1])


@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def cumulative_sum(carry: float, x: float) -> float:
    return carry + x


def define_rejected_programs() -> dict:
    """Define each program inside `pytest.raises`, as defining it is what raises."""

    def pure_copy(f: CellKField, h0: gtx.int32, h1: gtx.int32, v0: gtx.int32):
        copy_from_below(x=f, out=f, domain={Cell: (h0, h1), KDim: (v0, v0 + 1)})

    def pure_copy_negative_offset(f: CellKField, h0: gtx.int32, h1: gtx.int32, v0: gtx.int32):
        copy_from_above(x=f, out=f, domain={Cell: (h0, h1), KDim: (v0, v0 + 1)})

    def scaled(f: CellKField):
        scaled_from_below(f, out=f)

    def summed(f: CellKField, s: CellKField):
        sum_with_other(f, s, out=f)

    def nested(f: CellKField):
        scaled_copy_from_below(f, out=f)

    def multi_output_slot(f: CellKField, y: CellKField, g: CellKField):
        copy_and_scale(f, y, out=(f, g))

    def other_output_reads_shifted(f: CellKField, g: CellKField):
        shifted_and_incremented(f, out=(g, f))

    def after_conditional_return(f: CellKField, flag: bool):
        shifted_unless_flag(f, flag, out=f)

    def unstructured(f: EdgeField):
        neighbor_sum_of(f, out=f)

    def dynamic_offset(f: CellKField, offsets: KIntField):
        dynamically_shifted(f, offsets, out=f)

    return {
        "pure_copy": (pure_copy, "shifted by +1 along 'KDim'"),
        "pure_copy_negative_offset": (pure_copy_negative_offset, "shifted by -1 along 'KDim'"),
        "scaled": (scaled, "shifted by +1 along 'KDim'"),
        "summed": (summed, "shifted by +1 along 'KDim'"),
        "nested": (nested, "shifted by +1 along 'KDim'"),
        "multi_output_slot": (multi_output_slot, "shifted by +1 along 'KDim'"),
        "other_output_reads_shifted": (other_output_reads_shifted, "shifted by +1 along 'KDim'"),
        "after_conditional_return": (after_conditional_return, "shifted by +1 along 'KDim'"),
        "unstructured": (unstructured, "through 'E2E'"),
        "dynamic_offset": (dynamic_offset, "at a data-dependent offset"),
    }


REJECTED = list(define_rejected_programs().keys())


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: getattr(b, "name", "embedded"))
@pytest.mark.parametrize("case", REJECTED)
def test_inout_field_read_with_offset_is_rejected(case, backend):
    definition, offset = define_rejected_programs()[case]

    with pytest.raises(errors.InOutFieldReadWithOffsetError) as exc_info:
        gtx.program(definition, backend=backend, grid_type=gtx.GridType.UNSTRUCTURED)

    err = exc_info.value
    assert err.field_name == "f"
    assert err.notes[0] == f"'f' is read {offset}."
    # the caret points at the `f` in `out=...`
    assert err.location is not None and err.location.filename == __file__
    assert err.related and all(loc.filename == __file__ for loc, _ in err.related)


def test_inout_field_read_with_offset_location():
    def program_with_tuple_out(f: CellKField, y: CellKField, g: CellKField):
        copy_and_scale(f, y, out=(f, g))

    with pytest.raises(errors.InOutFieldReadWithOffsetError) as exc_info:
        gtx.program(program_with_tuple_out)

    err = exc_info.value
    line = program_with_tuple_out.__code__.co_firstlineno + 1
    assert (err.location.line, err.location.column) == (line, 35)
    assert err.label == "written here"
    assert [(loc.line, loc.column, label) for loc, label in err.related] == [
        (line, 24, "read here")
    ]


def define_accepted_programs() -> dict:
    def pointwise(f: CellKField):
        incremented(f, out=f)

    def pointwise_after_conditional_return(f: CellKField, flag: bool):
        incremented_unless_flag(f, flag, out=f)

    def out_not_read(f: CellKField, g: CellKField, h0: gtx.int32, h1: gtx.int32, v0: gtx.int32):
        copy_from_below(x=f, out=g, domain={Cell: (h0, h1), KDim: (v0, v0 + 1)})

    def multi_output_passthrough(f: CellKField, y: CellKField, g: CellKField):
        passthrough_and_use(x=f, y=y, out=(g, f))

    def forwarded_and_used_shifted(f: CellKField, y: CellKField, g: CellKField, flag: bool):
        forward_unless_flag(f, y, flag, out=(f, g))

    def zero_offset(f: CellKField):
        zero_shift(f, out=f)

    def shifts_cancel_out(f: CellKField):
        shift_and_shift_back(f, out=f)

    def scan(f: CellKField):
        cumulative_sum(f, out=f)

    def shifted_in_earlier_statement(f: CellKField, g: CellKField):
        copy_from_below(f, out=g)
        incremented(g, out=f)

    return {
        "pointwise": pointwise,
        "pointwise_after_conditional_return": pointwise_after_conditional_return,
        "out_not_read": out_not_read,
        "multi_output_passthrough": multi_output_passthrough,
        "forwarded_and_used_shifted": forwarded_and_used_shifted,
        "zero_offset": zero_offset,
        "shifts_cancel_out": shifts_cancel_out,
        "scan": scan,
        "shifted_in_earlier_statement": shifted_in_earlier_statement,
    }


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: getattr(b, "name", "embedded"))
@pytest.mark.parametrize("case", list(define_accepted_programs().keys()))
def test_inout_field_read_pointwise_is_accepted(case, backend):
    gtx.program(define_accepted_programs()[case], backend=backend)
