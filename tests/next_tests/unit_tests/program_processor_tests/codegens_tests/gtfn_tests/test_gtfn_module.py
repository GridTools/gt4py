# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import tempfile
import pathlib
import os
import pickle
import copy
import diskcache


import gt4py.next as gtx
import gt4py.next.config
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import arguments, languages, stages, workflow, toolchain
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.type_system import type_translation


@pytest.fixture
def fencil_example():
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
                    im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                    im.literal("10", itir.INTEGER_INDEX_BUILTIN),
                ],
            )
        ],
    )
    fencil = itir.FencilDefinition(
        id="example",
        params=[im.sym(name, type_) for name, type_ in zip(("buf", "sc"), param_types)],
        function_definitions=[
            itir.FunctionDefinition(
                id="stencil",
                params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
                expr=im.literal("1", "float32"),
            )
        ],
        closures=[
            itir.StencilClosure(
                domain=domain,
                stencil=itir.SymRef(id="stencil"),
                output=itir.SymRef(id="buf"),
                inputs=[itir.SymRef(id="buf"), itir.SymRef(id="sc")],
            )
        ],
    )
    return fencil, params


def test_codegen(fencil_example):
    fencil, parameters = fencil_example
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


def test_hash_and_diskcache(fencil_example):
    fencil, parameters = fencil_example
    compilable_program = stages.CompilableProgram(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete_no_size(
            *parameters, **{"offset_provider": {}}
        ),
    )

    hash = gtfn.generate_stencil_source_hash_function(compilable_program)
    path = str(gt4py.next.config.BUILD_CACHE_DIR / gt4py.next.config.GTFN_SOURCE_CACHE_DIR)
    with diskcache.Cache(path) as cache:
        cache[hash] = compilable_program

    # check content of cash file
    with diskcache.Cache(path) as reopened_cache:
        assert hash in reopened_cache
        compilable_program_from_cache = reopened_cache[hash]
        assert compilable_program == compilable_program_from_cache
        del reopened_cache[hash]  # delete data

    # hash creation is deterministic
    assert hash == gtfn.generate_stencil_source_hash_function(compilable_program)
    assert hash == gtfn.generate_stencil_source_hash_function(compilable_program_from_cache)

    # hash is different if program changes
    altered_program = copy.deepcopy(compilable_program)
    altered_program.data.id = "example2"
    assert gtfn.generate_stencil_source_hash_function(
        compilable_program
    ) != gtfn.generate_stencil_source_hash_function(altered_program)


def test_gtfn_file_cache(fencil_example):
    fencil, parameters = fencil_example
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

    cached_gtfn_translation_step(
        compilable_program
    )  # run cached translation step once to populate cache
    assert bare_gtfn_translation_step(compilable_program) == cached_gtfn_translation_step(
        compilable_program
    )

    cache_key = gtfn.generate_stencil_source_hash_function(compilable_program)
    assert cache_key in cached_gtfn_translation_step.cache
    assert (
        bare_gtfn_translation_step(compilable_program)
        == cached_gtfn_translation_step.cache[cache_key]
    )
