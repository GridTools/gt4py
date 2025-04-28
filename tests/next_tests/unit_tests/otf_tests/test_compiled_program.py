# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest


from gt4py import next as gtx, eve
from gt4py.next import errors
from gt4py.next.otf import compiled_program
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.ffront import type_specifications as ts_ffront

from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import simple_mesh


def test_offset_provider_to_type_unsafe():
    mesh = simple_mesh(None)
    offset_provider = mesh.offset_provider

    compiled_program._offset_provider_to_type_unsafe_impl.cache_clear()

    compiled_program._offset_provider_to_type_unsafe(offset_provider)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 1
    compiled_program._offset_provider_to_type_unsafe(offset_provider)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 1
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().hits == 1

    offset_provider2 = {"V2E": mesh.offset_provider["V2E"]}
    compiled_program._offset_provider_to_type_unsafe(offset_provider2)
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().misses == 2
    assert compiled_program._offset_provider_to_type_unsafe_impl.cache_info().hits == 1


class SomeEnum(eve.IntEnum):
    FOO = 1


@pytest.mark.parametrize(
    "value, type_, expected",
    [
        (gtx.int32(1), ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (gtx.int64(1), ts.ScalarType(kind=ts.ScalarKind.INT64), gtx.int64(1)),
        (1, ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (True, ts.ScalarType(kind=ts.ScalarKind.BOOL), True),
        (False, ts.ScalarType(kind=ts.ScalarKind.BOOL), False),
        (SomeEnum.FOO, ts.ScalarType(kind=ts.ScalarKind.INT32), gtx.int32(1)),
        (
            (1, (2.0, gtx.float32(3.0))),
            ts.TupleType(
                types=[
                    ts.ScalarType(kind=ts.ScalarKind.INT32),
                    ts.TupleType(
                        types=[
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT64),
                            ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
                        ]
                    ),
                ]
            ),
            (gtx.float32(1), (gtx.float64(2.0), gtx.float32(3.0))),
        ),
    ],
)
def test_sanitize_static_args(value, type_, expected):
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={
                "testee": type_,
            },
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )

    result = compiled_program._sanitize_static_args(
        "testee_program", {"testee": value}, program_type
    )
    assert result == {"testee": expected}


def test_sanitize_static_args_non_scalar_type():
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={
                "foo": ts.FieldType(dims=[], dtype=ts.ScalarType(kind=ts.ScalarKind.INT32))
            },
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )
    with pytest.raises(
        errors.TypeError_,
        match="foo.*cannot be static",
    ):
        compiled_program._sanitize_static_args("foo_program", {"foo": gtx.int32(1)}, program_type)


def test_sanitize_static_args_wrong_type():
    program_type = ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=[],
            pos_or_kw_args={"foo": ts.ScalarType(kind=ts.ScalarKind.INT32)},
            kw_only_args={},
            returns=ts.VoidType(),
        )
    )
    with pytest.raises(errors.TypeError_, match="got 'int64'"):
        compiled_program._sanitize_static_args("foo_program", {"foo": gtx.int64(1)}, program_type)
