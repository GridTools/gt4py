# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import diskcache
import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import arguments, languages, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.type_system import type_translation

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    KDim,
    exec_alloc_descriptor,
)


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
                expr=im.call(im.call("as_fieldop")(itir.SymRef(id="stencil"), domain))(
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
        stages.CompilableProgram(
            data=fencil,
            args=arguments.CompileTimeArgs.from_concrete_no_size(
                *parameters, **{"offset_provider": {}}
            ),
        )
    )
    assert module.entry_point.name == fencil.id
    assert any(d.name == "gridtools_cpu" for d in module.library_deps)
    assert module.language is languages.CPP


def test_hash_and_diskcache(program_example, tmp_path):
    fencil, parameters = program_example
    compilable_program = stages.CompilableProgram(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete_no_size(
            *parameters, **{"offset_provider": {}}
        ),
    )
    hash = stages.fingerprint_compilable_program(compilable_program)

    with diskcache.Cache(tmp_path) as cache:
        cache[hash] = compilable_program

    # check content of cash file
    with diskcache.Cache(tmp_path) as reopened_cache:
        assert hash in reopened_cache
        compilable_program_from_cache = reopened_cache[hash]
        assert compilable_program == compilable_program_from_cache
        del reopened_cache[hash]  # delete data

    # hash creation is deterministic
    assert hash == stages.fingerprint_compilable_program(compilable_program)
    assert hash == stages.fingerprint_compilable_program(compilable_program_from_cache)

    # hash is different if program changes
    altered_program_id = copy.deepcopy(compilable_program)
    altered_program_id.data.id = "example2"
    assert stages.fingerprint_compilable_program(
        compilable_program
    ) != stages.fingerprint_compilable_program(altered_program_id)

    altered_program_offset_provider = copy.deepcopy(compilable_program)
    object.__setattr__(altered_program_offset_provider.args, "offset_provider", {"Koff": KDim})
    assert stages.fingerprint_compilable_program(
        compilable_program
    ) != stages.fingerprint_compilable_program(altered_program_offset_provider)

    altered_program_column_axis = copy.deepcopy(compilable_program)
    object.__setattr__(altered_program_column_axis.args, "column_axis", KDim)
    assert stages.fingerprint_compilable_program(
        compilable_program
    ) != stages.fingerprint_compilable_program(altered_program_column_axis)


def test_gtfn_file_cache(program_example):
    fencil, parameters = program_example
    compilable_program = stages.CompilableProgram(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete_no_size(
            *parameters, **{"offset_provider": {}}
        ),
    )
    cached_gtfn_translation_step = gtfn.GTFNBackendFactory(
        gpu=False, cached=True, otf_workflow__cached_translation=True
    ).executor.step.translation

    bare_gtfn_translation_step = gtfn.GTFNBackendFactory(
        gpu=False, cached=True, otf_workflow__cached_translation=False
    ).executor.step.translation

    cache_key = stages.fingerprint_compilable_program(compilable_program)

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


# TODO(egparedes): we should switch to use the cached backend by default and then remove this test
def test_gtfn_file_cache_whole_workflow(cartesian_case):
    if cartesian_case.backend != gtfn.run_gtfn:
        pytest.skip("Skipping backend.")
    cartesian_case.backend = gtfn.GTFNBackendFactory(
        gpu=False, cached=True, otf_workflow__cached_translation=True
    )

    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        field_tuple = (a, a)
        field_0 = field_tuple[0]
        field_1 = field_tuple[1]
        return field_0

    # first call: this generates the cache file
    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)
    # clearing the OTFCompileWorkflow cache such that the OTFCompileWorkflow step is executed again
    object.__setattr__(cartesian_case.backend.executor, "cache", {})
    # second call: the cache file is used
    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: a)
