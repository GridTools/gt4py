# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

from gt4py._core import filecache
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next import common, fingerprinting
from gt4py.next.otf import arguments, artifacts, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.type_system import type_specifications as ts
from gt4py.next import custom_layout_allocators as next_allocators

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case, cartesian_case_no_backend
from next_tests.integration_tests.cases_utils import (
    V2E,
    Edge,
    KDim,
    Vertex,
    exec_alloc_descriptor,
    skip_value_mesh,
)


def _make_cartesian_program() -> itir.Program:
    """Point-wise operation on a 1D cartesian field."""
    float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
    param_types = [ts.FieldType(dims=[gtx.Dimension("I")], dtype=float_type), float_type]

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
    return itir.Program(
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


def _make_unstructured_program() -> itir.Program:
    """Neighbor reduction over 'V2E'."""
    float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    vftype = ts.FieldType(dims=[Vertex], dtype=float_type)
    return itir.Program(
        id="example_unstructured",
        function_definitions=[],
        params=[
            im.sym("x", ts.FieldType(dims=[Edge], dtype=float_type)),
            im.sym("y", vftype),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("it")(im.reduce("plus", im.literal_from_value(1.0))(im.deref("it")))
                )(im.as_fieldop_neighbors(V2E.value, "x")),
                domain=im.get_field_domain(common.GridType.UNSTRUCTURED, "y", vftype.dims),
                target=itir.SymRef(id="y"),
            )
        ],
    )


def _make_compilable_program(
    program: itir.Program, offset_provider: common.OffsetProvider
) -> stages.CompilableProgramDef:
    return stages.CompilableProgramDef(
        data=program,
        args=arguments.CompileTimeArgs(
            args=tuple(param.type for param in program.params),
            kwargs={},
            offset_provider=offset_provider,
            column_axis=None,
            argument_descriptor_contexts={},
        ),
    )


@pytest.mark.parametrize("grid_type", [common.GridType.CARTESIAN, common.GridType.UNSTRUCTURED])
def test_codegen(grid_type):
    if grid_type == common.GridType.CARTESIAN:
        program = _make_cartesian_program()
        offset_provider = {}
    else:
        program = _make_unstructured_program()
        offset_provider = skip_value_mesh(None).offset_provider

    module = gtfn_module.translate_program_cpu(_make_compilable_program(program, offset_provider))

    assert isinstance(module, gtfn_module.GTFNProgramSource)
    assert module.grid_type == grid_type
    assert module.entry_point.name == program.id
    assert any(d.name == "gridtools_cpu" for d in module.library_deps)
    assert isinstance(module.code_spec, artifacts.CPPCodeSpec)


def test_hash_and_diskcache(tmp_path):
    compilable_program = _make_compilable_program(_make_cartesian_program(), offset_provider={})
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


def test_gtfn_file_cache():
    compilable_program = _make_compilable_program(_make_cartesian_program(), offset_provider={})
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
