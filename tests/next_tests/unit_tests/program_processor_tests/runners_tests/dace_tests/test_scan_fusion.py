# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend scan fusion preserves recurrence, shifted inputs and boundaries."""

import copy

import dace
import numpy as np
import pytest

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import pass_manager
from gt4py.next.program_processors.runners.dace import scan_fusion
from gt4py.next.program_processors.runners.dace.workflow import translation
from gt4py.next.program_processors.runners.dace.workflow.translation import DaCeTranslator
from gt4py.next.type_system import type_specifications as ts


IDim = common.Dimension("IDim")
K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
FTYPE = ts.FieldType(dims=[IDim, K], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))


def make_program(levels=17, forward=True, shift_axis=K):
    domain = im.domain("cartesian_domain", {IDim: (1, 3), K: (1, levels + 1)})
    offset = im.cartesian_offset(shift_axis, shift_axis)
    coefficient = im.as_fieldop(
        im.lambda_("x")(
            im.plus(im.multiplies_(0.5, im.deref("x")), im.deref(im.shift(offset, -1)("x")))
        ),
        domain,
    )("x")
    other = im.as_fieldop(
        im.lambda_("x")(im.plus(im.deref("x"), im.deref(im.shift(offset, 1)("x")))), domain
    )("x")
    scan = im.scan(
        im.lambda_("carry", "a", "b")(
            im.make_tuple(
                im.plus(im.multiplies_(0.5, im.tuple_get(0, "carry")), im.deref("a")),
                im.plus(im.tuple_get(1, "carry"), im.deref("b")),
            )
        ),
        forward,
        im.make_tuple(1.0, 2.0),
    )
    return ir.Program(
        id="scan_coefficient_fusion",
        function_definitions=[],
        declarations=[],
        params=[im.sym(name, FTYPE) for name in ("x", "y", "z")],
        body=[
            ir.SetAt(
                expr=im.as_fieldop(scan, domain)(coefficient, other),
                domain=im.make_tuple(domain, domain),
                target=im.make_tuple("y", "z"),
            )
        ],
    )


def translate(program, enabled, **extra_options):
    options = {"fuse_scan_inputs": enabled, **extra_options}
    graph = DaCeTranslator(
        device_type=core_defs.DeviceType.CPU,
        auto_optimize=True,
        auto_optimize_args=options,
        async_sdfg_call=False,
        unstructured_horizontal_has_unit_stride=False,
        use_metrics=False,
    ).generate_sdfg(program, offset_provider={}, column_axis=K)
    assert options == {"fuse_scan_inputs": enabled, **extra_options}
    return graph


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("levels", [2, 17, 120])
def test_compiled_scan_fusion(tmp_path, forward, levels, selected=None):
    selector = None if selected is None else lambda call, index: index == selected
    graph = translate(make_program(levels, forward), True, scan_input_selector=selector)
    assert len(
        [
            d
            for d in graph.arrays.values()
            if d.transient and isinstance(d, dace.data.Array) and len(d.shape) == 2
        ]
    ) == (0 if selected is None else 1)
    graph.build_folder = str(tmp_path / "build")
    rng = np.random.default_rng(42)
    x = rng.uniform(-1, 1, (4, levels + 2))
    y = np.full_like(x, -71.0)
    z = np.full_like(x, -71.0)
    expected_y, expected_z = y.copy(), z.copy()
    indices = range(1, levels + 1) if forward else range(levels, 0, -1)
    for i in range(1, 3):
        a, b = 1.0, 2.0
        for k in indices:
            a = 0.5 * a + (0.5 * x[i, k] + x[i, k - 1])
            b = b + (x[i, k] + x[i, k + 1])
            expected_y[i, k], expected_z[i, k] = a, b
    args = dict(x=x, y=y, z=z)
    for name in ("x", "y", "z"):
        for dim, size, stride in (("IDim", 4, levels + 2), ("K", levels + 2, 1)):
            args.update(
                {
                    f"__{name}_{dim}_range_0": 0,
                    f"__{name}_{dim}_range_1": size,
                    f"__{name}_{dim}_stride": stride,
                }
            )
    graph(**{key: value for key, value in args.items() if key in graph.arglist()})
    np.testing.assert_allclose(y, expected_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z, expected_z, rtol=1e-12, atol=1e-12)


def test_disabled_keeps_coefficient_arrays():
    graph = translate(make_program(), False)
    assert (
        len(
            [
                d
                for d in graph.arrays.values()
                if d.transient and isinstance(d, dace.data.Array) and len(d.shape) == 2
            ]
        )
        == 2
    )


def test_unsupported_horizontal_shift_is_unchanged():
    program = pass_manager.apply_fieldview_transforms(
        make_program(shift_axis=IDim), offset_provider={}
    )
    before = copy.deepcopy(program)
    assert scan_fusion.fuse_scan_inputs(program, offset_provider={}) == before
    assert program == before


def test_scan_body_has_no_shifts_after_fusion():
    program = pass_manager.apply_fieldview_transforms(make_program(), offset_provider={})
    before = copy.deepcopy(program)
    result = scan_fusion.fuse_scan_inputs(program, offset_provider={})
    scans = [node for node in result.pre_walk_values() if cpm.is_call_to(node, "scan")]
    assert scans
    assert not any(
        cpm.is_applied_shift(node) for scan in scans for node in scan.args[0].pre_walk_values()
    )
    assert program == before


def test_invalid_switch():
    with pytest.raises(ValueError, match="must be a boolean"):
        translate(make_program(), "yes")


def test_input_output_overlap_is_unchanged():
    program = make_program()
    program.body[0].target = im.make_tuple("x", "z")
    program = pass_manager.apply_fieldview_transforms(program, offset_provider={})
    before = copy.deepcopy(program)
    assert scan_fusion.fuse_scan_inputs(program, offset_provider={}) == before
    assert program == before


def test_unrelated_in_place_output_does_not_disable_scan_fusion():
    program = make_program()
    program.params.append(im.sym("unrelated", FTYPE))
    statement = program.body[0]
    domain = statement.domain.args[0]
    statement.expr = im.make_tuple(statement.expr, im.as_fieldop("deref", domain)("unrelated"))
    statement.target = im.make_tuple(statement.target, "unrelated")
    statement.domain = im.make_tuple(statement.domain, domain)
    graph = translate(program, True)
    assert not [
        d
        for d in graph.arrays.values()
        if d.transient and isinstance(d, dace.data.Array) and len(d.shape) == 2
    ]


def test_let_alias_does_not_hide_input_output_overlap():
    program = make_program()
    statement = program.body[0]
    statement.target = im.make_tuple("x", "z")

    # Bind the scan input through a let, retaining the original field as output.
    class RenameInput(eve.NodeTranslator):
        def visit_Lambda(self, node, **kwargs):
            return node

        def visit_SymRef(self, node, **kwargs):
            return im.ref("alias") if node.id == "x" else node

    statement.expr = im.let("alias", "x")(RenameInput().visit(statement.expr))
    program = pass_manager.apply_fieldview_transforms(program, offset_provider={})
    before = copy.deepcopy(program)
    assert scan_fusion.fuse_scan_inputs(program, offset_provider={}) == before
    assert program == before


def test_shared_upstream_coefficient_remains_outside_scan():
    program = make_program()
    statement = program.body[0]
    domain = statement.domain.args[0]
    shared = im.as_fieldop(im.lambda_("input")(im.multiplies_(0.5, im.deref("input"))), domain)("x")
    offset = im.cartesian_offset(K, K)
    statement.expr.args = [
        im.as_fieldop(
            im.lambda_("factor", "values")(
                im.multiplies_(im.deref("factor"), im.deref(im.shift(offset, distance)("values")))
            ),
            domain,
        )("shared", "x")
        for distance in (-1, 1)
    ]
    statement.expr = im.let("shared", shared)(statement.expr)
    graph = translate(program, True)
    assert (
        len(
            [
                d
                for d in graph.arrays.values()
                if d.transient and isinstance(d, dace.data.Array) and len(d.shape) == 2
            ]
        )
        == 1
    )


def make_chained_program():
    program = make_program()
    call = program.body[0].expr
    domain = call.fun.args[1]
    upstream = im.as_fieldop(im.lambda_("v")(im.multiplies_(7.0, im.deref("v"))), domain)("x")
    call.args[0] = im.as_fieldop(im.lambda_("v")(im.plus(3.0, im.deref("v"))), domain)(upstream)
    return pass_manager.apply_fieldview_transforms(program, offset_provider={})


@pytest.mark.parametrize("selected", [0, 1])
def test_selected_input_only_and_no_recursive_producer_fusion(selected):
    program = make_chained_program()
    before = copy.deepcopy(program)
    original = program.body[0].expr
    calls = []

    def select(call, index):
        calls.append((copy.deepcopy(call), index))
        return index == selected

    result = scan_fusion.fuse_scan_inputs(program, offset_provider={}, input_selector=select)
    call = next(
        node
        for node in result.pre_walk_values()
        if cpm.is_applied_as_fieldop(node) and cpm.is_call_to(node.fun.args[0], "scan")
    )
    assert original.args[1 - selected] in call.args
    assert original.args[selected] not in call.args
    if selected == 0:
        assert original.args[0].args[0] in call.args
    assert len(calls) == 2
    assert all(node == original for node, _ in calls)
    assert program == before


def test_default_does_not_recursively_fuse_upstream():
    program = make_chained_program()
    upstream = copy.deepcopy(program.body[0].expr.args[0].args[0])
    result = scan_fusion.fuse_scan_inputs(program, offset_provider={})
    call = next(
        node
        for node in result.pre_walk_values()
        if cpm.is_applied_as_fieldop(node) and cpm.is_call_to(node.fun.args[0], "scan")
    )
    assert upstream in call.args


def test_selector_can_reject_all_inputs():
    program = make_chained_program()
    before = copy.deepcopy(program)
    result = scan_fusion.fuse_scan_inputs(
        program, offset_provider={}, input_selector=lambda call, index: False
    )
    assert result == before
    assert program == before


def test_selector_reaches_pass_through_backend():
    graph = translate(make_program(), True, scan_input_selector=lambda call, index: index == 0)
    assert (
        len(
            [
                d
                for d in graph.arrays.values()
                if d.transient and isinstance(d, dace.data.Array) and len(d.shape) == 2
            ]
        )
        == 1
    )


def test_invalid_selector():
    with pytest.raises(ValueError, match="must be callable"):
        translate(make_program(), True, scan_input_selector=(0,))


@pytest.mark.parametrize("scope", ["immediate", "field_operator"])
def test_fieldview_pipeline_runs_before_and_after_fusion(monkeypatch, scope):
    original = pass_manager.apply_fieldview_transforms
    calls = []

    def record(program, **kwargs):
        calls.append(kwargs)
        return original(program, **kwargs)

    monkeypatch.setattr(pass_manager, "apply_fieldview_transforms", record)
    monkeypatch.setattr(translation.itir_transforms, "apply_fieldview_transforms", record)
    program = make_scoped_program() if scope == "field_operator" else make_program()
    translate(program, True, scan_fusion_scope=scope)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[1]["offset_provider"] == {}
    assert "use_max_domain_range_on_unstructured_shift" in calls[1]


@pytest.mark.parametrize("selected", [0, 1])
def test_selected_input_compiled_values(tmp_path, selected):
    test_compiled_scan_fusion(tmp_path, True, 17, selected=selected)


def make_scoped_program():
    program = make_program()
    call = program.body[0].expr
    offset = im.cartesian_offset(K, K)
    wrapper = ir.FunctionDefinition(
        id="scan_wrapper",
        params=[im.sym("a"), im.sym("b")],
        expr=im.as_fieldop(call.fun.args[0])("a", "b"),
    )
    upstream = ir.FunctionDefinition(
        id="prepare_wind",
        params=[im.sym("values")],
        expr=im.as_fieldop(im.lambda_("v")(im.multiplies_(13.75, im.deref("v"))))("values"),
    )
    gamma = im.as_fieldop(im.lambda_("v")(im.multiplies_(0.5, im.deref("v"))))("values")
    a = im.as_fieldop(
        im.lambda_("g", "v")(im.multiplies_(im.deref("g"), im.deref(im.shift(offset, -1)("v"))))
    )("gamma", "values")
    b = im.as_fieldop(im.lambda_("g", "w")(im.plus(im.deref("g"), im.deref("w"))))("gamma", "wind")
    producer = ir.FunctionDefinition(
        id="coefficient_stage",
        params=[im.sym("values"), im.sym("wind")],
        expr=im.let("gamma", gamma)(im.call("scan_wrapper")(a, b)),
    )
    program.function_definitions = [wrapper, upstream, producer]
    program.body[0].expr = im.call("coefficient_stage")("x", im.call("prepare_wind")("x"))
    return program


def test_field_operator_scope_preserves_upstream_boundary():
    program = make_scoped_program()
    before = copy.deepcopy(program)
    prepared = scan_fusion.normalize_scan_producers(program, offset_provider={})
    normalized = pass_manager.apply_fieldview_transforms(prepared, offset_provider={})
    result = scan_fusion.fuse_scan_inputs(normalized, offset_provider={})
    scans = [
        n
        for n in result.pre_walk_values()
        if cpm.is_applied_as_fieldop(n) and cpm.is_call_to(n.fun.args[0], "scan")
    ]
    assert len(scans) == 1
    scan = scans[0]
    assert "13.75" not in str(scan.fun.args[0])
    assert any("13.75" in str(arg) for arg in scan.args)
    assert "0.5" in str(scan.fun.args[0])
    assert not any("gamma" in str(arg) for arg in scan.args)
    assert program == before


def test_scoped_preparation_does_not_duplicate_other_scans():
    program = make_scoped_program()
    stage = program.function_definitions[-1]
    stage.expr = im.let("wind", im.call("scan_wrapper")("values", "values"))(
        im.make_tuple(
            im.tuple_get(0, "wind"),
            im.tuple_get(1, "wind"),
        )
    )
    # A scan result used twice is a boundary, not an arithmetic let to expand.
    prepared = scan_fusion.normalize_scan_producers(program, offset_provider={})
    calls = [
        n
        for n in prepared.function_definitions[-1].expr.pre_walk_values()
        if cpm.is_call_to(n, "scan_wrapper")
    ]
    assert len(calls) == 1


def test_scoped_selection_and_alias_guard():
    prepared = scan_fusion.normalize_scan_producers(make_scoped_program(), offset_provider={})
    normalized = pass_manager.apply_fieldview_transforms(prepared, offset_provider={})
    original_scan = next(
        n
        for n in normalized.pre_walk_values()
        if cpm.is_applied_as_fieldop(n) and cpm.is_call_to(n.fun.args[0], "scan")
    )
    result = scan_fusion.fuse_scan_inputs(
        normalized, offset_provider={}, input_selector=lambda call, index: index == 0
    )
    scan = next(
        n
        for n in result.pre_walk_values()
        if cpm.is_applied_as_fieldop(n) and cpm.is_call_to(n.fun.args[0], "scan")
    )
    assert original_scan.args[1] in scan.args
    normalized.body[0].target = im.make_tuple("x", "z")
    before = copy.deepcopy(normalized)
    assert scan_fusion.fuse_scan_inputs(normalized, offset_provider={}) == before


@pytest.mark.parametrize("selected", [None, 0])
def test_scoped_compiled_values(tmp_path, selected):
    selector = None if selected is None else lambda call, index: index == selected
    graph = translate(
        make_scoped_program(),
        True,
        scan_fusion_scope="field_operator",
        scan_input_selector=selector,
    )
    graph.build_folder = str(tmp_path / "build")
    rng = np.random.default_rng(52)
    x = rng.uniform(-1, 1, (4, 19))
    y = np.full_like(x, -71.0)
    z = np.full_like(x, -71.0)
    expected_y, expected_z = y.copy(), z.copy()
    for i in range(1, 3):
        a, b = 1.0, 2.0
        for k in range(1, 18):
            gamma = 0.5 * x[i, k]
            a = 0.5 * a + gamma * x[i, k - 1]
            b = b + (gamma + 13.75 * x[i, k])
            expected_y[i, k], expected_z[i, k] = a, b
    args = dict(x=x, y=y, z=z)
    for name in ("x", "y", "z"):
        for dim, size, stride in (("IDim", 4, 19), ("K", 19, 1)):
            args.update(
                {
                    f"__{name}_{dim}_range_0": 0,
                    f"__{name}_{dim}_range_1": size,
                    f"__{name}_{dim}_stride": stride,
                }
            )
    graph(**{key: value for key, value in args.items() if key in graph.arglist()})
    np.testing.assert_allclose(y, expected_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(z, expected_z, rtol=1e-12, atol=1e-12)


def test_invalid_fusion_scope():
    with pytest.raises(ValueError, match="scan_fusion_scope"):
        translate(make_program(), True, scan_fusion_scope="recursive")


def test_stage_normalization_accepts_identity_fieldop():
    program = make_scoped_program()
    stage = program.function_definitions[-1]
    stage.expr.fun.expr.args[0] = im.as_fieldop("deref")(stage.expr.fun.expr.args[0])
    prepared = scan_fusion.normalize_scan_producers(program, offset_provider={})
    normalized = pass_manager.apply_fieldview_transforms(prepared, offset_provider={})
    result = scan_fusion.fuse_scan_inputs(normalized, offset_provider={})
    scan = next(
        n
        for n in result.pre_walk_values()
        if cpm.is_applied_as_fieldop(n) and cpm.is_call_to(n.fun.args[0], "scan")
    )
    assert "0.5" in str(scan.fun.args[0])
