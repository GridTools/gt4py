# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
import gt4py.next.config
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.otf import arguments, languages, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
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


def test_transformation_caching(fencil_example):
    program, _ = fencil_example
    args = dict(
        program=program,
        offset_provider={},
        column_axis=gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL),
    )

    # test cache file written is what the function returns
    with tempfile.TemporaryDirectory() as cache_dir:
        try:
            prev_cache_dir = gt4py.next.config.BUILD_CACHE_DIR
            gt4py.next.config.BUILD_CACHE_DIR = pathlib.Path(cache_dir)

            cache_file_path = gtfn_module._generate_stencil_source_cache_file_path(**args)
            assert not os.path.exists(cache_file_path)
            stencil_source = gtfn_module.translate_program_cpu.generate_stencil_source(**args)
            assert os.path.exists(cache_file_path)
            with open(cache_file_path, "rb") as f:
                stencil_source_from_cache = pickle.load(f)
            assert stencil_source == stencil_source_from_cache
        except Exception as e:
            raise e
        finally:
            gt4py.next.config.BUILD_CACHE_DIR = prev_cache_dir

    # test cache file is deterministic
    assert gtfn_module._generate_stencil_source_cache_file_path(
        **args
    ) == gtfn_module._generate_stencil_source_cache_file_path(**args)

    # test cache file changes for a different program
    altered_program = copy.deepcopy(program)
    altered_program.id = "example2"
    assert gtfn_module._generate_stencil_source_cache_file_path(
        **args
    ) != gtfn_module._generate_stencil_source_cache_file_path(
        **(args | {"program": altered_program})
    )
