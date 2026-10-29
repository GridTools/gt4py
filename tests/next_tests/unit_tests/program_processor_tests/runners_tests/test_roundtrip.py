# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

import gt4py.next as gtx
from gt4py.next import backend as next_backend
from gt4py.next.ffront import foast_to_gtir, foast_to_past, past_to_itir
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import arguments, stages
from gt4py.next.program_processors.runners import roundtrip


@pytest.mark.parametrize("debug_order", [(False, True), (True, False)])
def test_generate_source_ignores_debug_flag(debug_order, monkeypatch):
    monkeypatch.setattr(roundtrip, "_SOURCE_CACHE", {})
    domain = im.call("cartesian_domain")(im.named_range(itir.AxisLiteral(value="D"), 0, 1))
    program = itir.Program(
        id="testee",
        function_definitions=[],
        params=[itir.Sym(id="out")],
        declarations=[],
        body=[
            itir.SetAt(expr=im.as_fieldop("deref")(), domain=domain, target=itir.SymRef(id="out"))
        ],
    )
    transform_calls = []

    def transforms(ir, *, offset_provider):
        transform_calls.append(ir)
        return ir

    sources = [
        roundtrip._generate_source(
            program,
            debug=debug,
            use_embedded=True,
            offset_provider={},
            transforms=transforms,
        )
        for debug in debug_order
    ]

    assert sources[0] == sources[1]
    assert sources[0][1] == "testee"
    # The source is generated once and shared by both debug modes.
    assert len(transform_calls) == 1
    assert len(roundtrip._SOURCE_CACHE) == 1


@pytest.mark.parametrize("debug_order", [(False, True), (True, False)])
def test_load_module_caches_by_debug_flag(debug_order, monkeypatch):
    monkeypatch.setattr(roundtrip, "_MODULE_CACHE", {})
    source_code = "VALUE = 42\n"

    modules = {debug: roundtrip._load_module(source_code, debug) for debug in debug_order}
    try:
        assert modules[False] is not modules[True]
        assert modules[False].VALUE == modules[True].VALUE == 42
        # Only the debug module is backed by a real '.py' file.
        assert not hasattr(modules[False], "__file__")
        assert modules[True].__file__.endswith(".py")
        assert os.path.isfile(modules[True].__file__)

        for debug in debug_order:
            assert roundtrip._load_module(source_code, debug) is modules[debug]
    finally:
        os.remove(modules[True].__file__)


def _former_gtir_frontend() -> next_backend.Transforms:
    """The frontend `roundtrip.gtir` used to build for itself (before ADR 0029)."""
    return next_backend.Transforms(
        past_to_itir=past_to_itir.past_to_gtir_factory(),
        foast_to_itir=foast_to_gtir.foast_to_gtir_factory(cached=True),
        field_view_op_to_prog=foast_to_past.operator_to_program_factory(
            foast_to_itir_step=foast_to_gtir.foast_to_gtir_factory()
        ),
    )


def test_gtir_uses_the_default_frontend_without_changing_the_lowering():
    IDim = gtx.Dimension("IDim")

    @gtx.field_operator
    def double(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
        return a + a

    @gtx.program
    def double_program(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
        double(a, out=out)

    a = gtx.zeros({IDim: 4})
    out = gtx.zeros({IDim: 4})
    default_frontend = next_backend.Toolchain(roundtrip.gtir).frontend
    assert default_frontend is next_backend.DEFAULT_FRONTEND

    # A field operator is lowered as the program wrapping it, so `out` is positional.
    compile_time_args = arguments.CompileTimeArgs.from_concrete(a, out, offset_provider={})
    for definition_stage in [double_program.definition_stage, double.definition_stage]:
        inp = stages.ConcreteProgramDef(definition=definition_stage, args=compile_time_args)
        lowered = default_frontend(inp).definition
        assert isinstance(lowered, itir.Program)
        assert lowered == _former_gtir_frontend()(inp).definition
