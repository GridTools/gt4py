# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the translation stage of the dace backend workflow."""

import pytest

import re
import uuid
from unittest import mock

dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common, fingerprinting
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import arguments as otf_arguments, toolchain as otf_toolchain
from gt4py.next.program_processors.runners.dace.workflow import (
    translation as dace_wf_translation,
    common as dace_wf_common,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.cases_utils import (
    V2E,
    Edge,
    IDim,
    Vertex,
    skip_value_mesh,
)

from dace import nodes as dace_nodes


FLOAT_TYPE = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
IFTYPE = ts.FieldType(dims=[IDim], dtype=FLOAT_TYPE)
EFTYPE = ts.FieldType(dims=[Edge], dtype=FLOAT_TYPE)
VFTYPE = ts.FieldType(dims=[Vertex], dtype=FLOAT_TYPE)


@pytest.fixture(
    params=[
        pytest.param(core_defs.DeviceType.CPU),
        pytest.param(core_defs.DeviceType.CUDA, marks=[pytest.mark.requires_gpu]),
        pytest.param(core_defs.DeviceType.ROCM, marks=[pytest.mark.requires_gpu]),
    ],
    ids=["CPU", "CUDA", "ROCM"],
)
def device_type(request) -> str:
    return request.param


def _translate_gtir_to_sdfg(
    ir: itir.Program,
    offset_provider: gtx_common.OffsetProvider,
    device_type: core_defs.DeviceType,
    auto_optimize: bool,
    use_metrics: bool = False,
    max_concurrent_gpu_streams: int = 0,
    async_sdfg_call: bool = True,
) -> dace.SDFG:
    with dace.config.set_temporary("cache", value="hash"):
        # we use the SDFG hash in build cache to avoid clashes between CPU and GPU SDFGs
        return dace_wf_translation.DaCeTranslator(
            device_type=device_type,
            auto_optimize=auto_optimize,
            auto_optimize_args=None,
            async_sdfg_call=async_sdfg_call,
            unstructured_horizontal_has_unit_stride=False,
            use_metrics=use_metrics,
            max_concurrent_gpu_streams=max_concurrent_gpu_streams,
        ).generate_sdfg(ir, offset_provider=offset_provider, column_axis=None)


@pytest.mark.parametrize("has_unit_stride", [False, True])
@pytest.mark.parametrize("disable_field_origin", [False, True])
def test_find_constant_symbols(has_unit_stride, disable_field_origin):
    SKIP_VALUE_MESH = skip_value_mesh(None)

    ir = itir.Program(
        id="find_constant_symbols_sdfg",
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=EFTYPE),
            itir.Sym(id="y", type=VFTYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(im.reduce("plus", im.literal_from_value(1.0))(im.deref("it")))
                )(im.as_fieldop_neighbors(V2E.value, "x")),
                domain=im.get_field_domain(gtx_common.GridType.UNSTRUCTURED, "y", VFTYPE.dims),
                target=itir.SymRef(id="y"),
            )
        ],
    )

    sdfg = _translate_gtir_to_sdfg(
        ir=ir,
        offset_provider=SKIP_VALUE_MESH.offset_provider,
        device_type=core_defs.DeviceType.CPU,
        auto_optimize=False,
    )

    constant_symbols = dace_wf_translation.find_constant_symbols(
        ir=ir,
        sdfg=sdfg,
        offset_provider_type=SKIP_VALUE_MESH.offset_provider_type,
        disable_field_origin_on_program_arguments=disable_field_origin,
        unstructured_horizontal_has_unit_stride=has_unit_stride,
    )

    expected = {}
    if has_unit_stride:
        expected |= {
            "__x_Edge_stride": 1,
            "__y_Vertex_stride": 1,
            "__gt_conn_V2E_source_stride": 1,
        }
    if disable_field_origin:
        expected |= {
            "__x_Edge_range_0": 0,
            "__y_Vertex_range_0": 0,
        }
    assert constant_symbols == expected


def _make_simple_field_operator_compilable_program() -> otf_toolchain.ConcreteArtifact:
    """Return a compilable program wrapping a minimal GTIR field operator."""
    ir = itir.Program(
        id="simple_field_operator",
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="y", type=IFTYPE),
        ],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus")("x", 1.0),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", IFTYPE.dims),
                target=itir.SymRef(id="y"),
            ),
        ],
    )
    return otf_toolchain.ConcreteArtifact(
        data=ir,
        args=otf_arguments.CompileTimeArgs(
            args=tuple(param.type for param in ir.params),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={},
        ),
    )


def _increment_sdfg_guids(sdfg: dace.SDFG) -> None:
    """Increment the `guid` value of every SDFG element by one."""
    if hasattr(sdfg, "guid"):
        guid = uuid.UUID(str(sdfg.guid))
        sdfg.guid = str(uuid.UUID(int=guid.int + 1))
    for state in sdfg.states():
        if hasattr(state, "guid"):
            guid = uuid.UUID(str(state.guid))
            state.guid = str(uuid.UUID(int=guid.int + 1))
        for node in state.nodes():
            if hasattr(node, "guid"):
                guid = uuid.UUID(str(node.guid))
                node.guid = str(uuid.UUID(int=guid.int + 1))


def test_translation_source_code_invariant_under_guid_change():
    """SDFG `guid` changes must not alter the translation cache key.

    `DaCeTranslator.__call__` serializes the SDFG via `serialize_sdfg_as_json`,
    which drops `guid` values, before storing the SDFG JSON in
    mocks `build_sdfg_from_gtir` so that the second lowering returns the same
    SDFG with all `guid` values incremented by one, and verifies that the
    resulting `source_code` strings are identical.
    """
    compilable_program = _make_simple_field_operator_compilable_program()

    translator = dace_wf_translation.DaCeTranslator(
        device_type=core_defs.DeviceType.CPU,
        auto_optimize=False,
        auto_optimize_args=None,
        async_sdfg_call=True,
        unstructured_horizontal_has_unit_stride=False,
        use_metrics=False,
        max_concurrent_gpu_streams=0,
    )

    # Keep a reference to the real implementation so the mock can return the base
    # SDFG from the real implementation on the first call, then a guid-shifted clone.
    real_build_sdfg = dace_wf_translation.gtx_dace_lowering.build_sdfg_from_gtir
    base_sdfg: dace.SDFG | None = None
    call_count = 0

    def _build_sdfg_from_gtir_with_guid_change(*args: object, **kwargs: object) -> dace.SDFG:
        nonlocal base_sdfg, call_count
        call_count += 1
        if call_count == 1:
            base_sdfg = real_build_sdfg(*args, **kwargs)
            return base_sdfg
        assert base_sdfg is not None
        modified_sdfg = dace.SDFG.from_json(base_sdfg.to_json())
        _increment_sdfg_guids(modified_sdfg)
        assert modified_sdfg.to_json() != base_sdfg.to_json()
        return modified_sdfg

    with mock.patch.object(
        dace_wf_translation.gtx_dace_lowering,
        "build_sdfg_from_gtir",
        side_effect=_build_sdfg_from_gtir_with_guid_change,
    ):
        first_source = translator(compilable_program)
        second_source = translator(compilable_program)

    # different object identities, same content
    assert first_source.source_code is not second_source.source_code
    assert first_source.source_code == second_source.source_code

    assert first_source is not second_source
    first_source_fingerprint = fingerprinting.strict_fingerprinter(first_source)
    second_source_fingerprint = fingerprinting.strict_fingerprinter(second_source)
    assert first_source_fingerprint == second_source_fingerprint


@pytest.mark.parametrize("async_sdfg_call", [False, True], ids=["BLOCKING", "ASYNC"])
def test_translation_for_default_stream(async_sdfg_call: bool):
    """When max_concurrent_gpu_streams == 0, the SDFG either runs async or synchronizes on the default stream."""
    ir = itir.Program(
        id="testee",
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="y", type=IFTYPE),
        ],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus")("x", 1.0),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", IFTYPE.dims),
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = _translate_gtir_to_sdfg(
        ir=ir,
        offset_provider={},
        device_type=core_defs.DeviceType.CUDA,
        auto_optimize=False,
        async_sdfg_call=async_sdfg_call,
    )
    assert dace_wf_common.SDFG_ARG_EXTERNAL_SYNC_STREAM not in sdfg.symbols

    state_names = {s.label for s in sdfg.states()}

    if async_sdfg_call:
        assert "sync_entry" not in state_names
        assert "sync_exit" not in state_names
    else:
        assert "sync_entry" in state_names
        assert "sync_exit" in state_names

        entry_tasklets = [
            n
            for s in sdfg.states()
            for n in s.nodes()
            if isinstance(n, dace_nodes.Tasklet) and n.label == "sync_entry_tlet"
        ]
        exit_tasklets = [
            n
            for s in sdfg.states()
            for n in s.nodes()
            if isinstance(n, dace_nodes.Tasklet) and n.label == "sync_exit_tlet"
        ]
        assert len(entry_tasklets) == 1
        assert len(exit_tasklets) == 1
        entry_code = entry_tasklets[0].code.as_string
        exit_code = exit_tasklets[0].code.as_string
        assert "cudaStreamSynchronize" in entry_code
        assert "cudaStreamSynchronize" in exit_code
        assert "cudaStreamDefault" in entry_code
        assert "cudaStreamDefault" in exit_code


@pytest.mark.parametrize("async_sdfg_call", [False, True], ids=["BLOCKING", "ASYNC"])
def test_translation_adds_stream_sync_tasklets_for_multi_stream(async_sdfg_call: bool):
    """When max_concurrent_gpu_streams > 0, the SDFG contains entry/exit sync tasklets."""
    ir = itir.Program(
        id="testee",
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=IFTYPE),
            itir.Sym(id="y", type=IFTYPE),
        ],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus")("x", 1.0),
                domain=im.get_field_domain(gtx_common.GridType.CARTESIAN, "y", IFTYPE.dims),
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = _translate_gtir_to_sdfg(
        ir=ir,
        offset_provider={},
        device_type=core_defs.DeviceType.CUDA,
        auto_optimize=False,
        max_concurrent_gpu_streams=4,
        async_sdfg_call=async_sdfg_call,
    )
    assert dace_wf_common.SDFG_ARG_EXTERNAL_SYNC_STREAM in sdfg.symbols

    state_names = {s.label for s in sdfg.states()}
    assert "sync_entry" in state_names
    assert "sync_exit" in state_names

    entry_tasklets = [
        n
        for s in sdfg.states()
        for n in s.nodes()
        if isinstance(n, dace_nodes.Tasklet) and n.label == "sync_entry_tlet"
    ]
    exit_tasklets = [
        n
        for s in sdfg.states()
        for n in s.nodes()
        if isinstance(n, dace_nodes.Tasklet) and n.label == "sync_exit_tlet"
    ]
    assert len(entry_tasklets) == 1
    assert len(exit_tasklets) == 1
    entry_code = entry_tasklets[0].code.as_string
    exit_code = exit_tasklets[0].code.as_string
    assert "cudaStreamWaitEvent" in entry_code
    assert "cudaEventRecord" in exit_code
    assert dace_wf_common.SDFG_ARG_EXTERNAL_SYNC_STREAM in entry_code

    if async_sdfg_call:
        assert "cudaStreamSynchronize" not in entry_code
        assert "cudaStreamSynchronize" not in exit_code
    else:
        assert "cudaStreamSynchronize" in entry_code
        assert "cudaStreamSynchronize" in exit_code
