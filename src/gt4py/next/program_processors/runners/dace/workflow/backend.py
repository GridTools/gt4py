# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Final, ParamSpec

from gt4py.next import backend
from gt4py.next.otf import artifacts, recipes, workflow
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
    factory as gtx_wfdfactory,
    translation as gtx_wfdtranslation,
)


@dataclasses.dataclass(frozen=True)
class DaCeLoader:
    """
    Loader that injects an external workspace into the loaded program.

    The workspace is owned by the caller, not by the backend; it is installed
    onto the program wrapper before its first call, so that it is used when the
    SDFG argument vector is constructed.
    """

    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = None

    def __call__(self, artifact: artifacts.CompilationArtifact) -> artifacts.ExecutableProgram:
        program = artifacts.load_artifact(artifact)
        assert isinstance(program, gtx_wfddecoration.DaCeDecoratedProgram)
        program.set_external_workspace(self.external_workspace or {})
        return program


def is_dace_backend(backend_: Any) -> bool:
    """
    Check whether `backend_` is a compiled backend translating programs to SDFGs.

    The check looks at the translation step of the standard compile pipeline
    (through a `CachedStep`), not at the backend name, so renamed and customized
    DaCe backends are recognized too.

    Args:
        backend_: Any object, typically a `Backend`.

    Returns:
        `True` if `backend_` is a `CompiledBackend` whose compiler is the standard
        compile pipeline with a `DaCeTranslator` translation step.
    """
    if not isinstance(backend_, backend.CompiledBackend):
        return False
    compiler = backend_.compiler
    if not isinstance(compiler, recipes.OTFCompileWorkflow):
        return False
    translation: Any = compiler.translation
    if isinstance(translation, workflow.CachedStep):
        translation = translation.step
    return isinstance(translation, gtx_wfdtranslation.DaCeTranslator)


#: Keyword arguments of the flat-keyword `make_dace_backend` removed in ADR 0029,
#: mapped to where the setting lives now.
_REMOVED_MAKE_DACE_BACKEND_KEYWORDS: Final[dict[str, str]] = {
    "gpu": "DaCeConfig(gpu=...)",
    "auto_optimize": "DaCeConfig(auto_optimize=...)",
    "external_workspace": "DaCeConfig(external_workspace=...)",
    "unstructured_horizontal_has_unit_stride": (
        "DaCeConfig(unstructured_horizontal_has_unit_stride=...)"
    ),
    "optimization_args": "make_dace_translator(auto_optimize_args=...)",
    "async_sdfg_call": "make_dace_translator(async_sdfg_call=...)",
    "use_metrics": "make_dace_translator(use_metrics=...)",
    "use_zero_origin": "make_dace_translator(disable_field_origin_on_program_arguments=...)",
    "use_max_domain_range_on_unstructured_shift": (
        "make_dace_translator(use_max_domain_range_on_unstructured_shift=...)"
    ),
}

_P = ParamSpec("_P")


def _reject_removed_make_dace_backend_call(
    builder: Callable[_P, backend.CompiledBackend],
) -> Callable[_P, backend.CompiledBackend]:
    """Turn calls in the shape of the removed flat-keyword `make_dace_backend` into a clear error."""

    @functools.wraps(builder)
    def checked_builder(*args: _P.args, **kwargs: _P.kwargs) -> backend.CompiledBackend:
        removed = [name for name in kwargs if name in _REMOVED_MAKE_DACE_BACKEND_KEYWORDS]
        if removed or (args and not isinstance(args[0], gtx_wfdfactory.DaCeConfig | None)):
            moved = "".join(
                f"\n  - '{name}' -> {_REMOVED_MAKE_DACE_BACKEND_KEYWORDS[name]}" for name in removed
            )
            raise TypeError(
                "'make_dace_backend' takes a 'DaCeConfig' and step builders; the flat-keyword"
                " signature 'make_dace_backend(gpu, auto_optimize=..., ...)' was removed (ADR 0029)."
                " Call 'make_dace_backend(DaCeConfig(gpu=..., auto_optimize=...),"
                " translation=functools.partial(make_dace_translator, ...))' instead."
                + (f" Where the given arguments go:{moved}" if moved else "")
            )
        return builder(*args, **kwargs)

    return checked_builder


@_reject_removed_make_dace_backend_call
def make_dace_backend(
    cfg: gtx_wfdfactory.DaCeConfig | None = None,
    /,
    *,
    name_postfix: str = "",
    translation: gtx_wfdfactory.DaCeTranslationBuilder = gtx_wfdfactory.make_dace_translator,
    bindings: gtx_wfdfactory.DaCeBindingsBuilder = gtx_wfdfactory.make_dace_bindings,
    compilation: gtx_wfdfactory.DaCeCompilationBuilder = gtx_wfdfactory.make_dace_compiler,
) -> backend.CompiledBackend:
    """
    Build a compiled DaCe backend.

    Args:
        cfg: The backend configuration. Defaults to `DaCeConfig()`.
        name_postfix: Appended to the generated name. The generated name encodes
            only the device and whether auto-optimization is on; pass a postfix
            to tell other variants apart in metrics and logs.
        translation: Builder of the translation step, see
            `make_dace_compile_workflow`.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The configured backend.

    Raises:
        TypeError: If called with the arguments of the removed flat-keyword
            signature (`make_dace_backend(gpu, auto_optimize=..., ...)`).
    """
    if cfg is None:
        cfg = gtx_wfdfactory.DaCeConfig()

    return backend.CompiledBackend(
        name=f"run_dace_{cfg.device_name}{'_opt' if cfg.auto_optimize else ''}{name_postfix}",
        compiler=gtx_wfdfactory.make_dace_compile_workflow(
            cfg, translation=translation, bindings=bindings, compilation=compilation
        ),
        allocator=cfg.make_allocator(),
        loader=DaCeLoader(cfg.external_workspace),
    )


run_dace_cpu = make_dace_backend(gtx_wfdfactory.DaCeConfig(auto_optimize=True))
run_dace_cpu_noopt = make_dace_backend(gtx_wfdfactory.DaCeConfig(auto_optimize=False))

run_dace_gpu = make_dace_backend(gtx_wfdfactory.DaCeConfig(gpu=True, auto_optimize=True))
run_dace_gpu_noopt = make_dace_backend(gtx_wfdfactory.DaCeConfig(gpu=True, auto_optimize=False))
