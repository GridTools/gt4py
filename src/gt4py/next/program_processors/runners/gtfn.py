# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import pathlib
from typing import Any, TypedDict

import numpy as np

import gt4py._core.definitions as core_defs
from gt4py.next import backend, common, config, field_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.instrumentation import metrics
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import artifacts, recipes, stages
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import cache, compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


def convert_arg(arg: Any) -> Any:
    # Note: this function is on the hot path and needs to have minimal overhead.
    if (origin := getattr(arg, "__gt_origin__", None)) is not None:
        # `Field` is the most likely case, we use `__gt_origin__` as the property is needed anyway
        # and (currently) uniquely identifies a `NDArrayField` (which is the only supported `Field`)
        assert isinstance(arg, nd_array_field.NdArrayField)
        return arg.ndarray, origin
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if isinstance(arg, np.bool_):
        # nanobind does not support implicit conversion of `np.bool` to `bool`
        return bool(arg)
    # TODO(havogt): if this function still appears in profiles,
    # we should avoid going through the previous isinstance checks for detecting a scalar.
    # E.g. functools.cache on the arg type, returning a function that does the conversion
    return arg


def convert_args(
    inp: artifacts.ExecutableProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> artifacts.ExecutableProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.OffsetProviderElem],
        out: Any = None,
    ) -> None:
        # Note: this function is on the hot path and needs to have minimal overhead.
        if out is not None:
            args = (*args, out)
        converted_args = (convert_arg(arg) for arg in args)
        conn_args = extract_connectivity_args(offset_provider, device)

        opt_kwargs: dict[str, Any] = {}
        if collect_metrics := metrics.is_level_enabled(metrics.PERFORMANCE):
            # If we are collecting metrics, we need to add the `exec_info` argument
            # to the `inp` call, which will be used to collect performance metrics.
            exec_info: dict[str, float] = {}
            opt_kwargs["exec_info"] = exec_info

        # generate implicit domain size arguments only if necessary, using `iter_size_args()`
        inp(
            *converted_args,
            *conn_args,
            **opt_kwargs,
        )

        if collect_metrics:
            metrics.add_sample_to_current_source(
                metrics.COMPUTE_METRIC, exec_info["run_cpp_duration"]
            )

    return decorated_program


def extract_connectivity_args(
    offset_provider: dict[str, common.OffsetProviderElem], device: core_defs.DeviceType
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # Note: this function is on the hot path and needs to have minimal overhead.
    zero_origin = (0, 0)
    assert all(hasattr(conn, "ndarray") for conn in offset_provider.values())
    # Note: the order here needs to agree with the order of the generated bindings.
    # This is currently true only because when hashing offset provider dicts,
    # the keys' order is taken into account. Any modification to the hashing
    # of offset providers may break this assumption here.
    assert all(
        common.is_neighbor_table(conn) and field_utils.verify_device_field_type(conn, device)
        for conn in offset_provider.values()
        if hasattr(conn, "ndarray")
    )
    args: list[tuple[core_defs.NDArrayObject, tuple[int, ...]]] = [
        (conn.ndarray, zero_origin)
        for conn in offset_provider.values()
        if common.is_neighbor_table(conn)
    ]

    return args


@dataclasses.dataclass(frozen=True)
class GTFNCompilationArtifact(compiler.CPPCompilationArtifact):
    def load(self) -> artifacts.ExecutableProgram:
        return convert_args(super().load(), device=self.device_type)


@dataclasses.dataclass(frozen=True)
class GTFNCompiler(compiler.CPPCompiler):
    def _make_artifact(
        self, src_dir: pathlib.Path, module: pathlib.Path, entry_point_name: str
    ) -> GTFNCompilationArtifact:
        return GTFNCompilationArtifact(
            src_dir=src_dir,
            module=module,
            entry_point_name=entry_point_name,
            device_type=self.device_type,
        )


class GTFNTranslationOptions(TypedDict, total=False):
    """Step-local settings of `GTFNTranslationStep`; the device comes from the builder."""

    code_spec: artifacts.HeaderAndSourceCodeSpec | None
    enable_itir_transforms: bool
    symbolic_domain_sizes: dict[str, itir.Expr] | None
    use_max_domain_range_on_unstructured_shift: bool | None


class GTFNBuildSystemOptions(TypedDict, total=False):
    """Step-local settings of `CompiledbFactory`; the build type comes from the builder."""

    cmake_extra_flags: list[str]
    renew_compiledb: bool


class GTFNCompilationOptions(TypedDict, total=False):
    """Step-local settings of `GTFNCompiler`; device and cache lifetime come from the builder."""

    fingerprint_builder_factory: bool
    force_recompile: bool


def make_gtfn_compile_workflow(
    *,
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU,
    cached_translation: bool = False,
    cmake_build_type: config.CMakeBuildType | None = None,
    unstructured_horizontal_has_unit_stride: bool | None = None,
    translation: GTFNTranslationOptions | None = None,
    build_system: GTFNBuildSystemOptions | None = None,
    compilation: GTFNCompilationOptions | None = None,
) -> recipes.OTFCompileWorkflow:
    """
    Build the GTFN translation -> bindings -> compilation workflow.

    Settings shared by several steps are keyword arguments, forwarded to every
    step that needs them. The step-local settings of each step are passed as a
    dict, unpacked into the step's constructor; the shared settings are not
    part of these dicts, so they cannot be set for one step alone. To replace a
    whole step, use `dataclasses.replace` on the returned workflow.

    Args:
        device_type: The device the compiled program targets.
        cached_translation: Wrap the translation step in a persistent cache.
        cmake_build_type: Build type of the generated CMake project. Defaults
            to the value in `config`.
        unstructured_horizontal_has_unit_stride: Layout assumption of the
            bindings. Defaults to the value in `config`.
        translation: Step-local settings of the translation step.
        build_system: Step-local settings of the build system.
        compilation: Step-local settings of the compilation step.

    Returns:
        The composed compile workflow.
    """
    if cmake_build_type is None:
        cmake_build_type = config.CMAKE_BUILD_TYPE
    if unstructured_horizontal_has_unit_stride is None:
        unstructured_horizontal_has_unit_stride = config.UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE

    translation_step: stages.TranslationStep = gtfn_module.GTFNTranslationStep(
        device_type=device_type, **(translation or GTFNTranslationOptions())
    )
    if cached_translation:
        translation_step = cache.persistent_translation_cache(translation_step, "gtfn")

    return recipes.OTFCompileWorkflow(
        translation=translation_step,
        # `OTFCompileWorkflow` is not parameterized over the code spec, so its
        # `bindings` field is typed for `ProgramSource[Any]` while
        # `ExtensionGenerator` accepts only C++-like specs.
        bindings=nanobind.ExtensionGenerator(  # type: ignore[arg-type] # see comment above
            unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride
        ),
        compilation=GTFNCompiler(
            cache_lifetime=config.BUILD_CACHE_LIFETIME,
            builder_factory=compiledb.CompiledbFactory(
                cmake_build_type=cmake_build_type, **(build_system or GTFNBuildSystemOptions())
            ),
            device_type=device_type,
            **(compilation or GTFNCompilationOptions()),
        ),
    )


def make_gtfn_backend(
    *,
    gpu: bool = False,
    name_postfix: str = "",
    cached_translation: bool = True,
    cmake_build_type: config.CMakeBuildType | None = None,
    unstructured_horizontal_has_unit_stride: bool | None = None,
    translation: GTFNTranslationOptions | None = None,
    build_system: GTFNBuildSystemOptions | None = None,
    compilation: GTFNCompilationOptions | None = None,
) -> backend.Backend:
    """
    Build a GTFN backend.

    Args:
        gpu: Target the GPU instead of the CPU.
        name_postfix: Appended to the backend name, which must stay unique.
        cached_translation: Wrap the translation step in a persistent cache.
        cmake_build_type: See `make_gtfn_compile_workflow`.
        unstructured_horizontal_has_unit_stride: See `make_gtfn_compile_workflow`.
        translation: Step-local settings of the translation step.
        build_system: Step-local settings of the build system.
        compilation: Step-local settings of the compilation step.

    Returns:
        The configured backend.
    """
    device_type, allocator = backend.select_device(gpu)

    return backend.Backend(
        name=f"run_gtfn_{'gpu' if gpu else 'cpu'}{name_postfix}",
        executor=make_gtfn_compile_workflow(
            device_type=device_type,
            cached_translation=cached_translation,
            cmake_build_type=cmake_build_type,
            unstructured_horizontal_has_unit_stride=unstructured_horizontal_has_unit_stride,
            translation=translation,
            build_system=build_system,
            compilation=compilation,
        ),
        allocator=allocator,
        transforms=backend.DEFAULT_TRANSFORMS,
    )


run_gtfn = make_gtfn_backend()

run_gtfn_gpu = make_gtfn_backend(gpu=True)

run_gtfn_no_transforms = make_gtfn_backend(
    name_postfix="_no_transforms", translation={"enable_itir_transforms": False}
)
