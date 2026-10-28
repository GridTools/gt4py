# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Callable

from gt4py._core import definitions as core_defs
from gt4py.next import config, constructors, custom_layout_allocators as next_allocators
from gt4py.next.ffront import (
    foast_to_gtir,
    foast_to_past,
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, artifacts, stages, toolchain, workflow


def jit_to_aot_args(
    inp: arguments.JITArgs,
) -> arguments.CompileTimeArgs:
    return arguments.CompileTimeArgs.from_concrete(*inp.args, **inp.kwargs)


def adapted_jit_to_aot_args_factory() -> workflow.Workflow[
    stages.ConcreteProgramDef[stages.IRDefinitionT, arguments.JITArgs],
    stages.ConcreteProgramDef[stages.IRDefinitionT, arguments.CompileTimeArgs],
]:
    """Wrap `jit_to_aot` into a workflow adapter to fit into backend transform workflows."""
    return toolchain.ArgsOnlyAdapter(jit_to_aot_args)


@dataclasses.dataclass(frozen=True)
class Transforms(
    workflow.MultiWorkflow[
        stages.ConcreteProgramDef[stages.IRDefinitionT, stages.ArgsDefinitionT],
        stages.CompilableProgram,
    ]
):
    """
    Modular workflow for transformations with access to intermediates.

    The set and order of transformation steps depends on the input type.
    Thus this workflow can be applied to DSL field operator and program definitions,
    as well as their AST representations. Even to Iterator IR programs, although in that
    case it will be a no-op.

    The input to the workflow as well as each step must be a `CompilableProgram`. The arguments
    inside the `CompilableProgram` passed to the whole workflow may be concrete (`JITArgs`)
    or compile-time (`CompileTimeArgs`). The individual steps (apart from `.aotify_args`)
    require compile-time arguments. Some of the steps can work with an empty `CompileTimeArgs` instance.
    """

    aotify_args: workflow.Workflow[
        stages.ConcreteProgramDef[stages.IRDefinitionT, arguments.JITArgs],
        stages.ConcreteProgramDef[stages.IRDefinitionT, arguments.CompileTimeArgs],
    ] = dataclasses.field(default_factory=adapted_jit_to_aot_args_factory)

    func_to_foast: workflow.Workflow[
        ffront_stages.ConcreteDSLFieldOperatorDef, ffront_stages.ConcreteFOASTOperatorDef
    ] = dataclasses.field(default_factory=func_to_foast.adapted_func_to_foast_factory)

    func_to_past: workflow.Workflow[
        ffront_stages.ConcreteDSLProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=func_to_past.adapted_func_to_past_factory)

    foast_to_itir: workflow.Workflow[
        ffront_stages.ConcreteFOASTOperatorDef, itir.FunctionDefinition
    ] = dataclasses.field(default_factory=foast_to_gtir.adapted_foast_to_gtir_factory)

    field_view_op_to_prog: workflow.Workflow[
        ffront_stages.ConcreteFOASTOperatorDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=foast_to_past.operator_to_program_factory)

    past_lint: workflow.Workflow[
        ffront_stages.ConcretePASTProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=past_linters.adapted_linter_factory)

    field_view_prog_args_transform: workflow.Workflow[
        ffront_stages.ConcretePASTProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=past_process_args.transform_program_args_factory)

    past_to_itir: workflow.Workflow[
        ffront_stages.ConcretePASTProgramDef, stages.CompilableProgram
    ] = dataclasses.field(default_factory=past_to_itir.past_to_gtir_factory)

    def step_order(self, inp: stages.ConcreteProgramDef) -> list[str]:
        steps: list[str] = []
        if isinstance(inp.args, arguments.JITArgs):
            steps.append("aotify_args")
        match inp.definition:
            case ffront_stages.DSLFieldOperatorDef():
                steps.extend(
                    [
                        "func_to_foast",
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case ffront_stages.FOASTOperatorDef():
                steps.extend(
                    [
                        "field_view_op_to_prog",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case ffront_stages.DSLProgramDef():
                steps.extend(
                    [
                        "func_to_past",
                        "past_lint",
                        "field_view_prog_args_transform",
                        "past_to_itir",
                    ]
                )
            case ffront_stages.PASTProgramDef():
                steps.extend(["past_lint", "field_view_prog_args_transform", "past_to_itir"])
            case itir.Program():
                pass
            case _:
                raise ValueError("Unexpected input.")
        return steps


DEFAULT_TRANSFORMS: Transforms = Transforms()


@dataclasses.dataclass(frozen=True)
class Backend:
    """
    Abstract base of the objects passed as `backend=` to programs and operators.

    A backend decides how a program runs and where the buffers of its fields are
    allocated. This class is abstract: instantiate a concrete kind, such as
    `CompiledBackend`.

    Attributes:
        name: Human-readable label used in metrics, logs and diagnostics. It is
            expected to be unique among the backends used in a process; backends
            sharing a name are reported under the same label. gt4py does not use
            it to identify a backend: not as a cache key and not in fingerprints.
            Code that needs the kind of a backend should check its type, not its
            name.
        allocator: Allocates the buffers of fields used with this backend.
    """

    name: str
    allocator: constructors.Allocator

    def __post_init__(self) -> None:
        if type(self) is Backend:
            raise TypeError(
                "'Backend' is abstract: instantiate a concrete kind such as 'CompiledBackend'."
            )


@dataclasses.dataclass(frozen=True)
class CompiledBackend(Backend):
    """
    A backend that compiles programs before running them.

    Programs are lowered by the (global) frontend to a `CompilableProgram`, turned
    into a loadable `CompilationArtifact` by `compiler` (possibly in a worker
    process, see `config.BuildJobsMode`), and made directly callable by `loader`
    (always in the calling process). Backends needing backend-specific runtime
    data in the loaded program supply their own `loader`. Customize either step
    with `dataclasses.replace`.

    Attributes:
        allocator: The field buffer allocator for the device the compiled
            programs expect their buffers on.
        compiler: Compiles a lowered program into a loadable artifact.
        loader: Loads an artifact into a directly-callable program.
    """

    allocator: next_allocators.FieldBufferAllocatorProtocol
    compiler: workflow.Workflow[stages.CompilableProgram, artifacts.CompilationArtifact]
    # A plain `Callable`, not `workflow.Workflow`: the protocol names its
    # parameter `inp`, which would force every loader to use that name.
    loader: Callable[[artifacts.CompilationArtifact], artifacts.ExecutableProgram] = (
        dataclasses.field(default=artifacts.load_artifact)
    )

    @property
    def __gt_allocator__(self) -> next_allocators.FieldBufferAllocatorProtocol:
        return self.allocator


@dataclasses.dataclass(frozen=True)
class Toolchain:
    """
    Internal: the complete route from a program definition to an executable program.

    Composes the frontend, which transforms any supported program definition into
    a `CompilableProgram`, with a compiled backend, whose `compiler` turns it into
    a compilation artifact and whose `loader` makes that callable. Not part of the
    public API: gt4py builds it where a compilation happens.
    """

    backend: CompiledBackend
    frontend: workflow.Workflow[stages.ConcreteProgramDef, stages.CompilableProgram] = (
        DEFAULT_TRANSFORMS
    )

    def compile(
        self, definition: stages.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
    ) -> artifacts.ExecutableProgram:
        """Lower, compile and load `definition` in the calling thread."""
        compilable = self.frontend(
            stages.ConcreteProgramDef(definition=definition, args=compile_time_args)
        )
        return self.backend.loader(self.backend.compiler(compilable))


@dataclasses.dataclass(frozen=True)
class ToolchainConfig:
    """
    Settings that describe what a compiled toolchain builds for.

    A toolchain builder creates every step from one config, so steps that must
    agree on a setting (the target device, the build type, the cache lifetime,
    the data layout) read it from the same place and cannot drift apart.
    Settings that only tune how a single step does its job are not part of the
    config: they are keyword arguments of that step's builder.

    Defaults are read from `gt4py.next.config` when the config is created, not
    when this module is imported, so the toolchains built from a default config
    follow the current user configuration.
    """

    #: Target a GPU (the one CuPy was built for) instead of the CPU.
    gpu: bool = False
    #: Wrap the translation step in a persistent cache.
    cached_translation: bool = True
    cmake_build_type: config.CMakeBuildType = dataclasses.field(
        default_factory=lambda: config.CMAKE_BUILD_TYPE
    )
    cache_lifetime: config.BuildCacheLifetime = dataclasses.field(
        default_factory=lambda: config.BUILD_CACHE_LIFETIME
    )
    #: Assume unit stride in the horizontal dimension of unstructured fields.
    unstructured_horizontal_has_unit_stride: bool = dataclasses.field(
        default_factory=lambda: config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE
    )

    @property
    def device_type(self) -> core_defs.DeviceType:
        if self.gpu:
            return core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA
        return core_defs.DeviceType.CPU

    @property
    def device_name(self) -> str:
        """Device part of the toolchain name."""
        return "gpu" if self.gpu else "cpu"

    def make_allocator(self) -> next_allocators.FieldBufferAllocatorProtocol:
        if self.gpu:
            return next_allocators.StandardGPUFieldBufferAllocator()
        return next_allocators.StandardCPUFieldBufferAllocator()
