# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Final, Generic, NoReturn, Protocol, Self

import numpy as np

from gt4py._core import definitions as core_defs, ndarray_utils as core_ndarray_utils
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
from gt4py.next.otf import arguments, artifacts, stages, workflow


def jit_to_aot_args(
    inp: arguments.JITArgs,
) -> arguments.CompileTimeArgs:
    return arguments.CompileTimeArgs.from_concrete(*inp.args, **inp.kwargs)


@dataclasses.dataclass(frozen=True)
class Transforms:
    """
    Pipeline of the definition transformations, with access to intermediates.

    The set of transformation steps that run depends on the type of the input
    definition: DSL field operator and program definitions, their AST
    representations, and Iterator IR programs (for which the pipeline is a
    no-op) are all supported.

    The input is a `ProgramWithArgs` pair whose arguments may be concrete
    (`JITArgs`) or compile-time (`CompileTimeArgs`); `aotify_args` turns the
    former into the latter up front. Of the remaining steps only
    `field_view_op_to_prog`, `field_view_prog_args_transform` and
    `past_to_itir` consume the arguments at all; the others transform the bare
    definition. `stage_hook` is emitted after each executed step.
    Customization is composition-time: build a variant with
    `dataclasses.replace(transforms, past_lint=...)`.
    """

    aotify_args: workflow.Step[arguments.JITArgs, arguments.CompileTimeArgs] = jit_to_aot_args

    func_to_foast: workflow.Step[
        ffront_stages.DSLFieldOperatorDef, ffront_stages.FOASTOperatorDef
    ] = dataclasses.field(default_factory=func_to_foast.func_to_foast_factory)

    func_to_past: workflow.Step[ffront_stages.DSLProgramDef, ffront_stages.PASTProgramDef] = (
        dataclasses.field(default_factory=func_to_past.func_to_past_factory)
    )

    # Not part of the pipeline: `__call__` never runs this step. It is kept as a
    # configuration point for code that builds a variant frontend for the
    # internal `Toolchain` and needs to pin its FOAST -> ITIR lowering.
    foast_to_itir: workflow.Step[ffront_stages.FOASTOperatorDef, itir.FunctionDefinition] = (
        dataclasses.field(default_factory=foast_to_gtir.foast_to_gtir_factory)
    )

    field_view_op_to_prog: workflow.Step[
        ffront_stages.ConcreteFOASTOperatorDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=foast_to_past.operator_to_program_factory)

    past_lint: workflow.Step[ffront_stages.PASTProgramDef, ffront_stages.PASTProgramDef] = (
        dataclasses.field(default_factory=past_linters.linter_factory)
    )

    field_view_prog_args_transform: workflow.Step[
        ffront_stages.ConcretePASTProgramDef, ffront_stages.ConcretePASTProgramDef
    ] = dataclasses.field(default_factory=past_process_args.transform_program_args_factory)

    past_to_itir: workflow.Step[ffront_stages.ConcretePASTProgramDef, stages.CompilableProgram] = (
        dataclasses.field(default_factory=past_to_itir.past_to_gtir_factory)
    )

    def __call__(
        self, inp: stages.ConcreteProgramDef[stages.IRDefinitionT, stages.ArgsDefinitionT]
    ) -> stages.CompilableProgram:
        """
        Transform any supported program definition into a `CompilableProgram`.

        Which steps run is selected from the type of the input definition;
        `stage_hook` is emitted after each of them, carrying the very object
        handed on to the next step.

        Args:
            inp: A program definition in any supported stage, paired with the
                arguments it is compiled for. Concrete (`JITArgs`) arguments
                are turned into compile-time ones by `aotify_args` first.

        Returns:
            The Iterator IR program paired with its compile-time arguments. An
            input that already holds an `itir.Program` is passed through
            unchanged.

        Raises:
            ValueError: If the input definition is not one of the supported
                stages. Raised before any step runs, so nothing is emitted.
        """
        # The type guard runs before anything else, so an unsupported definition
        # is rejected before any step runs and before any `stage_hook` fires.
        if not isinstance(
            inp.definition,
            (
                ffront_stages.DSLFieldOperatorDef,
                ffront_stages.FOASTOperatorDef,
                ffront_stages.DSLProgramDef,
                ffront_stages.PASTProgramDef,
                itir.Program,
            ),
        ):
            raise ValueError("Unexpected input.")

        pair: stages.ConcreteProgramDef[stages.IRDefinitionT, arguments.CompileTimeArgs]
        if isinstance(inp.args, arguments.JITArgs):
            pair = workflow.ProgramWithArgs(inp.definition, self.aotify_args(inp.args))
            workflow.stage_hook("aotify_args", pair)
        else:
            pair = inp

        match pair.definition:
            case ffront_stages.DSLFieldOperatorDef() as dsl_operator:
                foast_pair = workflow.ProgramWithArgs(self.func_to_foast(dsl_operator), pair.args)
                workflow.stage_hook("func_to_foast", foast_pair)
                return self._transform_foast_operator(foast_pair)
            case ffront_stages.FOASTOperatorDef() as foast_operator:
                return self._transform_foast_operator(
                    workflow.ProgramWithArgs(foast_operator, pair.args)
                )
            case ffront_stages.DSLProgramDef() as dsl_program:
                past_pair = workflow.ProgramWithArgs(self.func_to_past(dsl_program), pair.args)
                workflow.stage_hook("func_to_past", past_pair)
                return self._transform_past_program(past_pair)
            case ffront_stages.PASTProgramDef() as past_program:
                return self._transform_past_program(
                    workflow.ProgramWithArgs(past_program, pair.args)
                )
            case itir.Program():
                # Nothing left to transform. This is the same object as `inp`
                # unless `aotify_args` had to rebuild the pair above.
                return pair
            case _:
                # Unreachable while the guard above lists exactly these stages;
                # it fails loudly if a stage is added to only one of the two.
                raise AssertionError(
                    f"Unhandled definition type '{type(pair.definition).__name__}'."
                )

    def _transform_foast_operator(
        self, operator: ffront_stages.ConcreteFOASTOperatorDef
    ) -> stages.CompilableProgram:
        past_pair = self.field_view_op_to_prog(operator)
        workflow.stage_hook("field_view_op_to_prog", past_pair)
        return self._transform_past_program(past_pair)

    def _transform_past_program(
        self, program: ffront_stages.ConcretePASTProgramDef
    ) -> stages.CompilableProgram:
        linted_pair = workflow.ProgramWithArgs(self.past_lint(program.definition), program.args)
        workflow.stage_hook("past_lint", linted_pair)
        args_pair = self.field_view_prog_args_transform(linted_pair)
        workflow.stage_hook("field_view_prog_args_transform", args_pair)
        compilable = self.past_to_itir(args_pair)
        workflow.stage_hook("past_to_itir", compilable)
        return compilable

    def step_order(self, inp: Any) -> NoReturn:
        """
        Tombstone of the removed input-dependent step-order hook.

        Raises:
            TypeError: Always. Overriding this method used to customize which
                steps run; it is now dead code, so it must fail loudly rather
                than silently restore the default behaviour.
        """
        raise TypeError(
            "'Transforms.step_order' was removed: the steps are now selected in"
            " 'Transforms.__call__'. Build a variant with 'dataclasses.replace'"
            " instead of overriding the step order (see ADR 0029)."
        )


DEFAULT_FRONTEND: Transforms = Transforms()


@typing.runtime_checkable
class DeviceConfigurable(Protocol):
    """A step that records the device it was configured for."""

    device_type: core_defs.DeviceType


def _declared_device_type(step: Any) -> core_defs.DeviceType | None:
    """
    Return the device a step declares it is configured for, if any.

    A step declares its device through a `device_type` attribute (the
    `DeviceConfigurable` protocol); `CachedStep` wrappers are looked
    through, since caching does not change the device of the wrapped step.

    Args:
        step: Any step. For a `CompilePipeline` this is the device its steps
            agree on.

    Returns:
        The declared device, or `None` if the step declares none.
    """
    while isinstance(step, workflow.CachedStep):
        step = step.step
    if isinstance(step, CompilePipeline):
        return step.device_type
    return step.device_type if isinstance(step, DeviceConfigurable) else None


@dataclasses.dataclass(frozen=True)
class CompilePipeline:
    """
    The standard three-step compile pipeline of the compiled backends.

    Turns a `CompilableProgram` into a loadable compilation artifact through
    source-code translation, bindings generation and compilation, emitting
    `stage_hook` after each step. Customization is composition-time: build a
    variant with `dataclasses.replace(pipeline, translation=...)`. The
    replacement is used as given: if the replaced step was wrapped in a
    `CachedStep`, the variant is uncached unless the new step is wrapped too.

    The steps that declare a device (see `device_type`) must all declare the
    same one. This is checked on construction, so it holds on every route to a
    pipeline, `dataclasses.replace` included. Steps that declare no device are
    not checked.
    """

    translation: stages.TranslationStep
    bindings: workflow.Step[artifacts.ProgramSource, artifacts.ExtensionSource]
    compilation: stages.CompilationStep

    def __post_init__(self) -> None:
        declared = self._declared_device_types()
        if len(set(declared.values())) > 1:
            steps = ", ".join(f"'{name}' for '{device.name}'" for name, device in declared.items())
            raise ValueError(
                f"The steps of a 'CompilePipeline' must target the same device, got {steps}."
            )

    def _declared_device_types(self) -> dict[str, core_defs.DeviceType]:
        steps = {
            "translation": self.translation,
            "bindings": self.bindings,
            "compilation": self.compilation,
        }
        return {
            name: device
            for name, step in steps.items()
            if (device := _declared_device_type(step)) is not None
        }

    @property
    def device_type(self) -> core_defs.DeviceType | None:
        """
        The device the steps are configured for.

        A step declares its device through a `device_type` attribute (the
        `DeviceConfigurable` protocol), looking through `CachedStep`.
        `None` if no step declares one.
        """
        return next(iter(self._declared_device_types().values()), None)

    def __call__(self, program: stages.CompilableProgram) -> artifacts.CompilationArtifact:
        """
        Run translation, bindings generation and compilation, in that order.

        `stage_hook` is emitted after each of the three steps.

        Args:
            program: The Iterator IR program paired with the compile-time
                arguments it is compiled for.

        Returns:
            The loadable artifact produced by the compilation step.
        """
        source = self.translation(program)
        workflow.stage_hook("translation", source)
        extension = self.bindings(source)
        workflow.stage_hook("bindings", extension)
        artifact = self.compilation(extension)
        workflow.stage_hook("compilation", artifact)
        return artifact


@dataclasses.dataclass(frozen=True)
class Backend:
    """
    Abstract base of the objects passed as `backend=` to programs and operators.

    A backend decides how a program runs and where the buffers of its fields are
    allocated. This class is abstract: instantiate a concrete kind, such as
    `CompiledBackend` or `EmbeddedBackend`.

    Backends are immutable configuration objects: a deep copy of a backend is the
    backend itself, so deep copies of programs share their backend, including any
    caller-owned state it holds (e.g. a DaCe external workspace).

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
                "'Backend' is abstract: instantiate a concrete kind such as 'CompiledBackend'"
                " or 'EmbeddedBackend'."
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        # Immutable configuration object: share its fields (array namespaces,
        # compile pipelines with caches, caller-owned buffers) instead of copying them.
        return self

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol | core_ndarray_utils.ArrayNamespace:
        """The allocator for fields used with this backend (unwrapping an allocator factory)."""
        allocator = self.allocator
        if next_allocators.is_field_allocator_factory(allocator):
            return allocator.__gt_allocator__
        return allocator


@dataclasses.dataclass(frozen=True)
class CompiledBackend(Backend, Generic[core_defs.DeviceTypeT]):
    """
    A backend that compiles programs before running them.

    Programs are lowered by the (global) frontend to a `CompilableProgram`, turned
    into a loadable `CompilationArtifact` by `compiler` (possibly in a worker
    process, see `config.BuildJobsMode`), and made directly callable by `loader`
    (always in the calling process). Backends needing backend-specific runtime
    data in the loaded program supply their own `loader`. Customize either step
    with `dataclasses.replace`.

    If both the allocator and the `compiler` declare a device (for a
    `CompilePipeline`, see `CompilePipeline.device_type`), they must be the
    same. This is checked on construction, so it also holds for a variant built
    with `dataclasses.replace`.

    Attributes:
        allocator: The field buffer allocator for the device the compiled
            programs expect their buffers on.
        compiler: Compiles a lowered program into a loadable artifact.
        loader: Loads an artifact into a directly-callable program.
    """

    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    compiler: workflow.Step[stages.CompilableProgram, artifacts.CompilationArtifact]
    loader: workflow.Step[artifacts.CompilationArtifact, artifacts.ExecutableProgram] = (
        dataclasses.field(default=artifacts.load_artifact)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not next_allocators.is_field_allocator(self.allocator):
            return
        allocator_device = self.allocator.__gt_device_type__
        compiler_device = _declared_device_type(self.compiler)
        if compiler_device is not None and allocator_device is not compiler_device:
            raise ValueError(
                f"Backend '{self.name}' allocates buffers on '{allocator_device.name}',"
                f" but its compiler targets '{compiler_device.name}'."
            )

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator


_NUMPY_NAMESPACE: Final = typing.cast(core_ndarray_utils.ArrayNamespace, np)


@dataclasses.dataclass(frozen=True)
class EmbeddedBackend(Backend):
    """
    A backend that runs the Python definition directly on fields, without lowering.

    Execution is the same for every instance: embedded operators dispatch on the
    array types of their arguments. The allocator only decides where gt4py
    allocates the buffers of fields for this backend, e.g. in
    `gtx.zeros(..., allocator=program)`. The instances live in
    `gt4py.next.program_processors.runners.embedded`.

    Attributes:
        allocator: The array namespace (e.g. `numpy`, `cupy`, `jax.numpy`) used
            to allocate field buffers. Defaults to NumPy.
    """

    allocator: core_ndarray_utils.ArrayNamespace = _NUMPY_NAMESPACE

    def __post_init__(self) -> None:
        super().__post_init__()
        if not core_ndarray_utils.is_array_namespace(self.allocator):
            raise TypeError(
                f"The allocator of embedded backend '{self.name}' must be an array namespace"
                f" (e.g. 'numpy'), got '{self.allocator}'."
            )

    @property
    def __gt_allocator__(self) -> core_ndarray_utils.ArrayNamespace:
        return self.allocator


@dataclasses.dataclass(frozen=True)
class Toolchain(Generic[core_defs.DeviceTypeT]):
    """
    Internal: the complete route from a program definition to an executable program.

    Composes the frontend, which transforms any supported program definition into
    a `CompilableProgram`, with a compiled backend, whose `compiler` turns it into
    a compilation artifact and whose `loader` makes that callable. Not part of the
    public API: gt4py builds it where a compilation happens.
    """

    backend: CompiledBackend[core_defs.DeviceTypeT]
    frontend: workflow.Step[stages.ConcreteProgramDef, stages.CompilableProgram] = DEFAULT_FRONTEND

    def compile(
        self, definition: stages.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
    ) -> artifacts.ExecutableProgram:
        """Lower, compile and load `definition` in the calling thread."""
        compilable = self.frontend(
            stages.ConcreteProgramDef(definition=definition, args=compile_time_args)
        )
        return self.backend.loader(self.backend.compiler(compilable))

    def translate(
        self, definition: stages.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
    ) -> artifacts.ProgramSource:
        """
        Run the frontend and the translation step only.

        This is the sanctioned partial run for stage inspection: the program
        definition goes through the full `frontend` pipeline and the
        `translation` step of the backend's compile pipeline, stopping before
        bindings generation and compilation. Per-call step options are
        deliberately not offered; a caller needing a variant translation step
        builds a variant backend with
        `dataclasses.replace(backend, compiler=dataclasses.replace(...))`.

        Args:
            definition: A program definition in any stage the frontend accepts.
            compile_time_args: Compile-time arguments for the frontend
                transforms and the translation step.

        Returns:
            The `ProgramSource` produced by the translation step.

        Raises:
            NotImplementedError: If the backend's `compiler` is not the standard
                `CompilePipeline` shape.
        """
        compiler = self.backend.compiler
        if not isinstance(compiler, CompilePipeline):
            raise NotImplementedError(
                f"Backend '{self.backend.name}' does not support partial runs: 'translate'"
                " requires the standard 'CompilePipeline' compile pipeline"
                " ('translation' / 'bindings' / 'compilation' steps), but this"
                f" backend's compiler is a '{type(compiler).__name__}'."
                " Monolithic compilers execute in a single step and produce no"
                " intermediate 'ProgramSource'."
            )
        source = compiler.translation(
            self.frontend(stages.ConcreteProgramDef(definition=definition, args=compile_time_args))
        )
        workflow.stage_hook("translation", source)
        return source


@dataclasses.dataclass(frozen=True)
class CompiledBackendConfig:
    """
    Settings that describe what a compiled backend builds for.

    A backend builder creates every step from one config, so steps that must
    agree on a setting (the target device, the build type, the cache lifetime,
    the data layout) read it from the same place and cannot drift apart.
    Settings that only tune how a single step does its job are not part of the
    config: they are keyword arguments of that step's builder.

    Defaults are read from `gt4py.next.config` when the config is created, not
    when this module is imported, so the backends built from a default config
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
        """Device part of the backend name."""
        return "gpu" if self.gpu else "cpu"

    def make_allocator(self) -> next_allocators.FieldBufferAllocatorProtocol:
        if self.gpu:
            return next_allocators.StandardGPUFieldBufferAllocator()
        return next_allocators.StandardCPUFieldBufferAllocator()
