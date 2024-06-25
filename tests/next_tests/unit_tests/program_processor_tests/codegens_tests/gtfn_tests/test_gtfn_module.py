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
import copy
import os.path
import pathlib
import pickle
import tempfile

import numpy as np
import pytest

import gt4py.next as gtx
import gt4py.next.config
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import languages, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.iterator.ir_utils import ir_makers as im


@pytest.fixture
def fencil_example():
    domain = itir.FunCall(
        fun=itir.SymRef(id="cartesian_domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value="X"),
                    im.literal("0", itir.INTEGER_INDEX_BUILTIN),
                    im.literal("10", itir.INTEGER_INDEX_BUILTIN),
                ],
            )
        ],
    )
    fencil = itir.FencilDefinition(
        id="example",
        params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
        function_definitions=[
            itir.FunctionDefinition(
                id="stencil",
                params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
                expr=im.literal("1", "float64"),
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
    IDim = gtx.Dimension("I")
    params = [gtx.as_field([IDim], np.empty((1,), dtype=np.float32)), np.float32(3.14)]
    return fencil, params


def test_codegen(fencil_example):
    fencil, parameters = fencil_example
    module = gtfn_module.translate_program_cpu(
        stages.ProgramCall(fencil, parameters, {"offset_provider": {}})
    )
    assert module.entry_point.name == fencil.id
    assert any(d.name == "gridtools_cpu" for d in module.library_deps)
    assert module.language is languages.Cpp


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
