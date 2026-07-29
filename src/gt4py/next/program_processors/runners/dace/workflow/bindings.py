# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from gt4py.next.otf import code_specs, stages


def _create_sdfg_bindings(
    program_source: stages.ProgramSource[code_specs.SDFGCodeSpec],
    bind_func_name: str,
) -> stages.BindingSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec]:
    """
    Creates a Python translation function to convert the GT4Py arguments list
    to the SDFG calling convention.

    Args:
        program_source: The json representation of the SDFG.
        bind_func_name: Name to use for the translation function.

    Returns:
        The Python code to convert call arguments from gt4py canonical form to the
        SDFG canonical form.
    """
    assert isinstance(program_source.source_code, tuple)
    assert len(program_source.source_code) == 2

    binding_code_generated_during_translation = program_source.source_code[0]
    return stages.BindingSource(binding_code_generated_during_translation, library_deps=tuple())


def bind_sdfg(
    inp: stages.ProgramSource[code_specs.SDFGCodeSpec],
    bind_func_name: str,
) -> stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec]:
    """
    Method to be used as workflow stage for generation of SDFG bindings.

    Refer to `_create_sdfg_bindings` documentation.
    """
    return stages.ExtensionSource(
        program_source=inp,
        binding_source=_create_sdfg_bindings(inp, bind_func_name),
    )
