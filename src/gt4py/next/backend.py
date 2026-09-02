# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Generic, NoReturn

from gt4py._core import definitions as core_defs
from gt4py.next import custom_layout_allocators as next_allocators
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
    # configuration point for downstream code (and for `roundtrip`, its only
    # in-repo writer) that needs to pin the FOAST -> ITIR lowering of a toolchain.
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


DEFAULT_TRANSFORMS: Transforms = Transforms()


@dataclasses.dataclass(frozen=True)
class CompilePipeline:
    """
    The standard three-step compile pipeline of the compiled backends.

    Turns a `CompilableProgram` into a loadable compilation artifact through
    source-code translation, bindings generation and compilation, emitting
    `stage_hook` after each step. Customization is composition-time: build a
    variant with `dataclasses.replace(pipeline, translation=...)`.
    """

    translation: stages.TranslationStep
    bindings: workflow.Step[artifacts.ProgramSource, artifacts.ExtensionSource]
    compilation: workflow.Step[artifacts.ExtensionSource, artifacts.CompilationArtifact]

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
class Toolchain(Generic[core_defs.DeviceTypeT]):
    """
    Complete pipeline from a program definition to an executable program.

    The `frontend` workflow transforms any supported program definition into a
    `CompilableProgram`, which the `backend` workflow then compiles into a
    loadable compilation artifact. The `loading` step turns that artifact into a
    directly-callable program; toolchains needing to inject backend-specific
    runtime data supply their own step here. The `allocator` describes the
    device the compiled program expects its buffers on.
    """

    name: str
    backend: workflow.Step[stages.CompilableProgram, artifacts.CompilationArtifact]
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    frontend: workflow.Step[stages.ConcreteProgramDef, stages.CompilableProgram]
    loading: workflow.Step[artifacts.CompilationArtifact, artifacts.ExecutableProgram] = (
        dataclasses.field(default=artifacts.load_artifact)
    )

    def compile(
        self, program: stages.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
    ) -> artifacts.ExecutableProgram:
        artifact = self.backend(
            self.frontend(stages.ConcreteProgramDef(definition=program, args=compile_time_args))
        )
        return self.loading(artifact)

    def translate(
        self, definition: stages.IRDefinitionT, compile_time_args: arguments.CompileTimeArgs
    ) -> artifacts.ProgramSource:
        """
        Run the frontend and the translation step only.

        This is the sanctioned partial run of the toolchain for stage
        inspection: the program definition goes through the full `frontend`
        pipeline and the `translation` step of the compile pipeline, stopping
        before bindings generation and compilation. Per-call step options are
        deliberately not offered; a caller needing a variant translation step
        builds a variant pipeline with `dataclasses.replace`.

        Args:
            definition: A program definition in any stage the frontend accepts.
            compile_time_args: Compile-time arguments for the frontend
                transforms and the translation step.

        Returns:
            The `ProgramSource` produced by the translation step.

        Raises:
            NotImplementedError: If this toolchain's `backend` is not the
                standard `CompilePipeline` pipeline shape.
        """
        if not isinstance(self.backend, CompilePipeline):
            raise NotImplementedError(
                f"Toolchain '{self.name}' does not support partial runs: 'translate'"
                " requires the standard 'CompilePipeline' compile pipeline"
                " ('translation' / 'bindings' / 'compilation' steps), but this"
                f" toolchain's backend is a '{type(self.backend).__name__}'."
                " Monolithic backends execute in a single step and produce no"
                " intermediate 'ProgramSource'."
            )
        source = self.backend.translation(
            self.frontend(stages.ConcreteProgramDef(definition=definition, args=compile_time_args))
        )
        workflow.stage_hook("translation", source)
        return source

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
