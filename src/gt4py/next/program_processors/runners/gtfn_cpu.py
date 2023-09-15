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

from typing import Any

import numpy.typing as npt

from gt4py.eve.utils import content_hash
from gt4py.next import common
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.otf import languages, recipes, stages, workflow
from gt4py.next.otf.binding import cpp_interface, nanobind
from gt4py.next.otf.compilation import cache, compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors import otf_compile_executor
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.type_system.type_translation import from_value


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if common.is_field(arg):
        arr = arg.ndarray
        origin = getattr(arg, "__gt_origin__", tuple([0] * len(arg.domain)))
        return arr, origin
    else:
        return arg


def convert_args(inp: stages.CompiledProgram) -> stages.CompiledProgram:
    def decorated_program(
        *args, offset_provider: dict[str, common.Connectivity | common.Dimension]
    ):
        converted_args = [convert_arg(arg) for arg in args]
        conn_args = extract_connectivity_args(offset_provider)
        return inp(
            *converted_args,
            *conn_args,
        )

    return decorated_program


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension]
) -> list[tuple[npt.NDArray, tuple[int, ...]]]:
    # note: the order here needs to agree with the order of the generated bindings
    args: list[tuple[npt.NDArray, tuple[int, ...]]] = []
    for name, conn in offset_provider.items():
        if isinstance(conn, common.Connectivity):
            if not isinstance(conn, common.NeighborTable):
                raise NotImplementedError(
                    "Only `NeighborTable` connectivities implemented at this point."
                )
            args.append((conn.table, tuple([0] * 2)))
        elif isinstance(conn, common.Dimension):
            pass
        else:
            raise AssertionError(
                f"Expected offset provider `{name}` to be a `Connectivity` or `Dimension`, "
                f"but got {type(conn).__name__}."
            )
    return args


def compilation_hash(otf_closure: stages.ProgramCall) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile."""
    offset_provider = otf_closure.kwargs["offset_provider"]
    return hash(
        (
            otf_closure.program,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            content_hash(tuple(from_value(arg) for arg in otf_closure.args)),
            id(offset_provider) if offset_provider else None,
            otf_closure.kwargs.get("column_axis", None),
        )
    )


GTFN_DEFAULT_TRANSLATION_STEP = gtfn_module.GTFNTranslationStep(
    cpp_interface.CPP_DEFAULT, enable_itir_transforms=True, use_imperative_backend=False
)

GTFN_DEFAULT_COMPILE_STEP = compiler.Compiler(
    cache_strategy=cache.Strategy.SESSION, builder_factory=compiledb.CompiledbFactory()
)


GTFN_DEFAULT_WORKFLOW = recipes.OTFCompileWorkflow(
    translation=GTFN_DEFAULT_TRANSLATION_STEP,
    bindings=nanobind.bind_source,
    compilation=GTFN_DEFAULT_COMPILE_STEP,
    decoration=convert_args,
)


run_gtfn = otf_compile_executor.OTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
](name="run_gtfn", otf_workflow=GTFN_DEFAULT_WORKFLOW)

run_gtfn_imperative = otf_compile_executor.OTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
](
    name="run_gtfn_imperative",
    otf_workflow=run_gtfn.otf_workflow.replace(
        translation=run_gtfn.otf_workflow.translation.replace(use_imperative_backend=True),
    ),
)

run_gtfn_cached = otf_compile_executor.CachedOTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
](
    name="run_gtfn_cached",
    otf_workflow=workflow.CachedStep(step=run_gtfn.otf_workflow, hash_function=compilation_hash),
)  # todo(ricoh): add API for converting an executor to a cached version of itself and vice versa


run_gtfn_with_temporaries = otf_compile_executor.OTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
](
    name="run_gtfn_with_temporaries",
    otf_workflow=run_gtfn.otf_workflow.replace(
        translation=run_gtfn.otf_workflow.translation.replace(lift_mode=LiftMode.FORCE_TEMPORARIES),
    ),
)
