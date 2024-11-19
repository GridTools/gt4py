# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from collections.abc import Callable, Mapping, Sequence
from inspect import currentframe, getframeinfo
from pathlib import Path
from typing import Any, Optional

import dace
from dace.sdfg import utils as sdutils
from dace.transformation.auto import auto_optimize as autoopt

import gt4py.next.iterator.ir as itir
from gt4py.next import common
from gt4py.next.iterator.transforms import (
    pass_manager_legacy as legacy_itir_transforms,
    program_to_fencil,
)
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.type_system import type_specifications as ts

from .itir_to_sdfg import ItirToSDFG


def preprocess_program(
    program: itir.FencilDefinition,
    offset_provider: Mapping[str, Any],
    lift_mode: legacy_itir_transforms.LiftMode,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    unroll_reduce: bool = False,
):
    node = legacy_itir_transforms.apply_common_transforms(
        program,
        common_subexpression_elimination=False,
        force_inline_lambda_args=True,
        lift_mode=lift_mode,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        temporary_extraction_heuristics=temporary_extraction_heuristics,
        unroll_reduce=unroll_reduce,
    )

    node = itir_type_inference.infer(node, offset_provider=offset_provider)

    if isinstance(node, itir.Program):
        fencil_definition = program_to_fencil.program_to_fencil(node)
        tmps = node.declarations
        assert all(isinstance(tmp, itir.Temporary) for tmp in tmps)
    else:
        raise TypeError(f"Expected 'Program', got '{type(node).__name__}'.")

    return fencil_definition, tmps


def build_sdfg_from_itir(
    program: itir.FencilDefinition,
    arg_types: Sequence[ts.TypeSpec],
    offset_provider: dict[str, Any],
    auto_optimize: bool = False,
    on_gpu: bool = False,
    column_axis: Optional[common.Dimension] = None,
    lift_mode: legacy_itir_transforms.LiftMode = legacy_itir_transforms.LiftMode.FORCE_INLINE,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    load_sdfg_from_file: bool = False,
    save_sdfg: bool = True,
    use_field_canonical_representation: bool = True,
) -> dace.SDFG:
    """Translate a Fencil into an SDFG.

    Args:
        program:             The Fencil that should be translated.
        arg_types:           Types of the arguments passed to the fencil.
        offset_provider:     The set of offset providers that should be used.
        auto_optimize:       Apply DaCe's `auto_optimize` heuristic.
        on_gpu:              Performs the translation for GPU, defaults to `False`.
        column_axis:         The column axis to be used, defaults to `None`.
        lift_mode:           Which lift mode should be used, defaults `FORCE_INLINE`.
        symbolic_domain_sizes: Used for generation of liskov bindings when temporaries are enabled.
        load_sdfg_from_file: Allows to read the SDFG from file, instead of generating it, for debug only.
        save_sdfg:           If `True`, the default the SDFG is stored as a file and can be loaded, this allows to skip the lowering step, requires `load_sdfg_from_file` set to `True`.
        use_field_canonical_representation: If `True`,  assume that the fields dimensions are sorted alphabetically.
    """

    sdfg_filename = f"_dacegraphs/gt4py/{program.id}.sdfg"
    if load_sdfg_from_file and Path(sdfg_filename).exists():
        sdfg: dace.SDFG = dace.SDFG.from_file(sdfg_filename)
        sdfg.validate()
        return sdfg

    # visit ITIR and generate SDFG
    program, tmps = preprocess_program(
        program, offset_provider, lift_mode, symbolic_domain_sizes, temporary_extraction_heuristics
    )
    sdfg_genenerator = ItirToSDFG(
        list(arg_types), offset_provider, tmps, use_field_canonical_representation, column_axis
    )
    sdfg = sdfg_genenerator.visit(program)
    if sdfg is None:
        raise RuntimeError(f"Visit failed for program {program.id}.")

    for nested_sdfg in sdfg.all_sdfgs_recursive():
        if not nested_sdfg.debuginfo:
            _, frameinfo = (
                warnings.warn(
                    f"{nested_sdfg.label} does not have debuginfo. Consider adding them in the corresponding nested sdfg.",
                    stacklevel=2,
                ),
                getframeinfo(currentframe()),  # type: ignore[arg-type]
            )
            nested_sdfg.debuginfo = dace.dtypes.DebugInfo(
                start_line=frameinfo.lineno, end_line=frameinfo.lineno, filename=frameinfo.filename
            )

    # TODO(edopao): remove `inline_loop_blocks` when DaCe transformations support LoopRegion construct
    sdutils.inline_loop_blocks(sdfg)

    # run DaCe transformations to simplify the SDFG
    sdfg.simplify()

    # run DaCe auto-optimization heuristics
    if auto_optimize:
        # TODO: Investigate performance improvement from SDFG specialization with constant symbols,
        #       for array shape and strides, although this would imply JIT compilation.
        symbols: dict[str, int] = {}
        device = dace.DeviceType.GPU if on_gpu else dace.DeviceType.CPU
        sdfg = autoopt.auto_optimize(sdfg, device, symbols=symbols, use_gpu_storage=on_gpu)
    elif on_gpu:
        autoopt.apply_gpu_storage(sdfg)

    if on_gpu:
        sdfg.apply_gpu_transformations()

    # Store the sdfg such that we can later reuse it.
    if save_sdfg:
        sdfg.save(sdfg_filename)

    return sdfg
