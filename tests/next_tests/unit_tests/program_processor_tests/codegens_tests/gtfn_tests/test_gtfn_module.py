# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

from gt4py._core import filecache
import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next import fingerprinting
from gt4py.next.otf import arguments, artifacts, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.type_system import type_specifications as ts, type_translation
from gt4py.next import custom_layout_allocators as next_allocators

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case, cartesian_case_no_backend
from next_tests.integration_tests.cases_utils import (
    IDim,
    KDim,
    exec_alloc_descriptor,
)


IKFloatField = gtx.Field[gtx.Dims[IDim, KDim], gtx.float64]


@pytest.fixture
def program_example():
    IDim = gtx.Dimension("I")
    params = [gtx.as_field([IDim], np.empty((1,), dtype=np.float32)), np.float32(3.14)]
    param_types = [type_translation.from_value(param) for param in params]

    domain = itir.FunCall(
        fun=itir.SymRef(id="cartesian_domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value="I"),
                    im.literal("0", builtins.INTEGER_INDEX_BUILTIN),
                    im.literal("10", builtins.INTEGER_INDEX_BUILTIN),
                ],
            )
        ],
    )
    program = itir.Program(
        id="example",
        params=[im.sym(name, type_) for name, type_ in zip(("buf", "sc"), param_types)],
        function_definitions=[
            itir.FunctionDefinition(
                id="stencil",
                params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
                expr=im.literal("1", "float32"),
            )
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(itir.SymRef(id="stencil"), domain)(
                    itir.SymRef(id="buf"), itir.SymRef(id="sc")
                ),
                domain=domain,
                target=itir.SymRef(id="buf"),
            )
        ],
    )
    return program, params


def test_codegen(program_example):
    fencil, parameters = program_example
    module = gtfn_module.translate_program_cpu(
        stages.CompilableProgramDef(
            data=fencil,
            args=arguments.CompileTimeArgs.from_concrete(*parameters, **{"offset_provider": {}}),
        )
    )
    assert module.entry_point.name == fencil.id
    assert any(d.name == "gridtools_cpu" for d in module.library_deps)
    assert isinstance(module.code_spec, artifacts.CPPCodeSpec)


def test_hash_and_diskcache(program_example, tmp_path):
    fencil, parameters = program_example
    compilable_program = stages.CompilableProgramDef(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete(*parameters, **{"offset_provider": {}}),
    )
    hash = fingerprinting.strict_fingerprinter(compilable_program)

    cache = filecache.FileCache(tmp_path)
    cache[hash] = compilable_program

    # check content of cash file
    reopened_cache = filecache.FileCache(tmp_path)
    assert hash in reopened_cache
    compilable_program_from_cache = reopened_cache[hash]
    assert compilable_program == compilable_program_from_cache
    del reopened_cache[hash]  # delete data

    # hash creation is deterministic
    assert hash == fingerprinting.strict_fingerprinter(compilable_program)
    assert hash == fingerprinting.strict_fingerprinter(compilable_program_from_cache)

    # hash is different if program changes
    altered_program_id = copy.deepcopy(compilable_program)
    altered_program_id.data.id = "example2"
    assert fingerprinting.strict_fingerprinter(
        compilable_program
    ) != fingerprinting.strict_fingerprinter(altered_program_id)

    altered_program_offset_provider = copy.deepcopy(compilable_program)
    object.__setattr__(altered_program_offset_provider.args, "offset_provider", {"Koff": KDim})
    assert fingerprinting.strict_fingerprinter(
        compilable_program
    ) != fingerprinting.strict_fingerprinter(altered_program_offset_provider)

    altered_program_column_axis = copy.deepcopy(compilable_program)
    object.__setattr__(altered_program_column_axis.args, "column_axis", KDim)
    assert fingerprinting.strict_fingerprinter(
        compilable_program
    ) != fingerprinting.strict_fingerprinter(altered_program_column_axis)


def test_gtfn_file_cache(program_example):
    fencil, parameters = program_example
    compilable_program = stages.CompilableProgramDef(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete(*parameters, **{"offset_provider": {}}),
    )
    cached_gtfn_translation_step = gtfn.make_gtfn_compile_workflow(
        gtfn.GTFNConfig(cached_translation=True)
    ).translation

    bare_gtfn_translation_step = gtfn.make_gtfn_compile_workflow(
        gtfn.GTFNConfig(cached_translation=False)
    ).translation

    cache_key = cached_gtfn_translation_step.cache_key(compilable_program)

    # ensure the actual cached step in the backend generates the cache item for the test
    if cache_key in (translation_cache := cached_gtfn_translation_step.cache):
        del translation_cache[cache_key]
    cached_gtfn_translation_step(compilable_program)
    assert bare_gtfn_translation_step(compilable_program) == cached_gtfn_translation_step(
        compilable_program
    )

    assert cache_key in cached_gtfn_translation_step.cache
    assert (
        bare_gtfn_translation_step(compilable_program)
        == cached_gtfn_translation_step.cache[cache_key]
    )


def _program_with_materialized_local():
    @gtx.field_operator
    def multiply_add(a: IKFloatField, b: IKFloatField) -> IKFloatField:
        return a * b + 1.0

    @gtx.field_operator
    def testee_op(a: IKFloatField, b: IKFloatField) -> IKFloatField:
        tmp = multiply_add(a, b)
        tmp = multiply_add(tmp, b)
        # the vertical shifts force `tmp` to be materialized, so its name survives lowering
        return tmp(KDim - 1) + tmp + tmp(KDim + 1)

    @gtx.program
    def testee(a: IKFloatField, b: IKFloatField, out: IKFloatField):
        testee_op(a, b, out=out, domain={IDim: (0, 4), KDim: (1, 3)})

    return testee, ("field", "field", "field")


def _program_with_scan_local():
    @gtx.scan_operator(axis=KDim, forward=True, init=0.0)
    def testee_scan(state: float, a: float) -> float:
        tmp = state + a
        return tmp * tmp

    @gtx.program
    def testee(a: IKFloatField, out: IKFloatField):
        testee_scan(a, out=out)

    return testee, ("field", "field")


def _program_with_if_stmt():
    @gtx.field_operator
    def testee_op(a: IKFloatField, condition: bool) -> IKFloatField:
        if condition:
            tmp = a + 1.0
        else:
            tmp = a * 2.0
        return tmp(KDim + 1) + tmp

    @gtx.program
    def testee(a: IKFloatField, condition: bool, out: IKFloatField):
        testee_op(a, condition, out=out, domain={IDim: (0, 4), KDim: (0, 3)})

    return testee, ("field", True, "field")


@pytest.mark.parametrize(
    "make_testee",
    [_program_with_materialized_local, _program_with_scan_local, _program_with_if_stmt],
    ids=["materialized_local", "scan_local", "if_stmt"],
)
def test_generated_source_is_ascii(make_testee):
    """
    Generated code must not contain the non-ASCII names of the frontend's SSA pass.

    The SSA pass renames `tmp` to e.g. `tmp\u141e0`. Compilers without support for
    extended identifiers (e.g. GCC < 10) reject such a name in C++ source.
    """
    testee, arg_kinds = make_testee()
    field = gtx.as_field([IDim, KDim], np.zeros((4, 4)))
    args = [field if kind == "field" else kind for kind in arg_kinds]

    module = gtfn_module.translate_program_cpu(
        stages.CompilableProgramDef(
            data=testee.gtir,
            args=arguments.CompileTimeArgs.from_concrete(
                *args, offset_provider={}, column_axis=KDim
            ),
        )
    )

    non_ascii_lines = [line for line in module.source_code.splitlines() if not line.isascii()]
    assert not non_ascii_lines


def test_replace_invalid_symbols():
    field_type = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    domain = im.domain(gtx.GridType.CARTESIAN, {IDim: (0, 10)})

    def make_program(lambda_params: tuple[str, str]) -> itir.Program:
        return itir.Program(
            id="testee",
            function_definitions=[],
            params=[im.sym("inp", field_type), im.sym("out", field_type)],
            declarations=[],
            body=[
                itir.SetAt(
                    expr=im.as_fieldop(
                        im.lambda_(*lambda_params)(
                            im.plus(im.deref(lambda_params[0]), im.deref(lambda_params[1]))
                        ),
                        domain,
                    )("inp", "inp"),
                    domain=domain,
                    target=im.ref("out"),
                )
            ],
        )

    # `tmp_0` is taken, so the C++ compatible spelling of `tmp\u141e0` has to be made unique
    testee = make_program(("tmp\u141e0", "tmp_0"))
    expected = make_program(("tmp_0_", "tmp_0"))

    assert gtfn_module.replace_invalid_symbols(testee) == expected


def test_replace_invalid_symbols_no_change():
    testee = itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.ref("inp"),
                domain=im.domain(gtx.GridType.CARTESIAN, {IDim: (0, 10)}),
                target=im.ref("out"),
            )
        ],
    )

    assert gtfn_module.replace_invalid_symbols(testee) is testee


def test_replace_invalid_symbols_keeps_params():
    # program parameters also name the parameters of the generated entry point
    testee = itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("inp\u03b1"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.ref("inp\u03b1"),
                domain=im.domain(gtx.GridType.CARTESIAN, {IDim: (0, 10)}),
                target=im.ref("out"),
            )
        ],
    )

    assert gtfn_module.replace_invalid_symbols(testee) is testee
