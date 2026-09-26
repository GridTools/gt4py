# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np

import gt4py._core.definitions as core_defs
from gt4py._core import filecache
from gt4py.next import backend, common, field_utils, fingerprinting
from gt4py.next.embedded import nd_array_field
from gt4py.next.instrumentation import metrics
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import artifacts, stages, workflow
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


@dataclasses.dataclass(frozen=True)
class GTFNConfig(backend.ToolchainConfig):
    """Settings shared by the steps of a GTFN toolchain, see `ToolchainConfig`."""


def make_gtfn_translation(
    cfg: GTFNConfig,
    /,
    *,
    enable_itir_transforms: bool = True,
    symbolic_domain_sizes: dict[str, itir.Expr] | None = None,
    use_max_domain_range_on_unstructured_shift: bool | None = None,
) -> gtfn_module.GTFNTranslationStep:
    """
    Build the GTIR -> C++ translation step.

    Args:
        cfg: The toolchain configuration.
        enable_itir_transforms: Run the GTIR transformation passes before code
            generation.
        symbolic_domain_sizes: Symbolic sizes of the domains, by dimension name.
        use_max_domain_range_on_unstructured_shift: See `GTFNTranslationStep`.

    Returns:
        The translation step, targeting `cfg.device_type`.
    """
    return gtfn_module.GTFNTranslationStep(
        device_type=cfg.device_type,
        enable_itir_transforms=enable_itir_transforms,
        symbolic_domain_sizes=symbolic_domain_sizes,
        use_max_domain_range_on_unstructured_shift=use_max_domain_range_on_unstructured_shift,
    )


def make_gtfn_bindings(
    cfg: GTFNConfig, /
) -> workflow.Step[artifacts.ProgramSource, artifacts.ExtensionSource]:
    """Build the step generating the nanobind bindings of the translated program."""
    return nanobind.ExtensionGenerator(
        unstructured_horizontal_has_unit_stride=cfg.unstructured_horizontal_has_unit_stride
    )


def make_gtfn_build_system(
    cfg: GTFNConfig,
    /,
    *,
    cmake_extra_flags: list[str] | None = None,
    renew_compiledb: bool = False,
) -> compiledb.CompiledbFactory:
    """
    Build the build-system project generator used by the GTFN compiler.

    Args:
        cfg: The toolchain configuration.
        cmake_extra_flags: Extra flags passed to CMake.
        renew_compiledb: Regenerate the compilation database even if one exists.

    Returns:
        A `CompiledbFactory` using `cfg.cmake_build_type`.
    """
    return compiledb.CompiledbFactory(
        cmake_build_type=cfg.cmake_build_type,
        cmake_extra_flags=cmake_extra_flags or [],
        renew_compiledb=renew_compiledb,
    )


def make_gtfn_compiler(
    cfg: GTFNConfig,
    /,
    *,
    build_system: Callable[[GTFNConfig], compiler.BuildSystemProjectGenerator] = (
        make_gtfn_build_system
    ),
    force_recompile: bool = False,
) -> GTFNCompiler:
    """
    Build the compilation step.

    Args:
        cfg: The toolchain configuration.
        build_system: Builder of the build-system project generator.
        force_recompile: Recompile even if a cached build exists.

    Returns:
        The compilation step, targeting `cfg.device_type`.
    """
    return GTFNCompiler(
        cache_lifetime=cfg.cache_lifetime,
        builder_factory=build_system(cfg),
        device_type=cfg.device_type,
        force_recompile=force_recompile,
    )


def make_gtfn_compile_workflow(
    cfg: GTFNConfig | None = None,
    /,
    *,
    translation: Callable[[GTFNConfig], stages.TranslationStep] = make_gtfn_translation,
    bindings: Callable[
        [GTFNConfig], workflow.Step[artifacts.ProgramSource, artifacts.ExtensionSource]
    ] = make_gtfn_bindings,
    compilation: Callable[
        [GTFNConfig], workflow.Step[artifacts.ExtensionSource, artifacts.CompilationArtifact]
    ] = make_gtfn_compiler,
) -> backend.CompilePipeline:
    """
    Build the GTFN translation -> bindings -> compilation workflow.

    Every step is created by a step builder that receives `cfg`, so all steps
    agree on the settings in it. To customize a step, pass a different
    builder: a `functools.partial` of the default one to change a
    step-local setting, e.g.
    `translation=functools.partial(make_gtfn_translation, enable_itir_transforms=False)`,
    or any callable taking the config to replace the step. The translation
    step is wrapped in the cache here, after its builder ran, so a custom
    translation step is cached like the default one.

    Args:
        cfg: The toolchain configuration. Defaults to `GTFNConfig()`.
        translation: Builder of the translation step.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The composed compile workflow.

    Raises:
        ValueError: If a step builder returns a step configured for a device
            other than `cfg.device_type`.
    """
    if cfg is None:
        cfg = GTFNConfig()

    translation_step = translation(cfg)
    workflow.check_device_agreement(translation_step, cfg.device_type, "GTFN translation step")
    compilation_step = compilation(cfg)
    workflow.check_device_agreement(compilation_step, cfg.device_type, "GTFN compilation step")

    if cfg.cached_translation:
        translation_step = workflow.CachedStep[
            stages.CompilableProgram, artifacts.ProgramSource, str
        ].persistent(
            translation_step,
            input_fingerprinter=fingerprinting.strict_fingerprinter,
            cache=filecache.FileCache(
                cache.get_translation_cache_folder(
                    cache.get_cache_base_path(cfg.cache_lifetime), "gtfn"
                )
            ),
        )

    return backend.CompilePipeline(
        translation=translation_step, bindings=bindings(cfg), compilation=compilation_step
    )


def make_gtfn_toolchain(
    cfg: GTFNConfig | None = None,
    /,
    *,
    name_postfix: str = "",
    translation: Callable[[GTFNConfig], stages.TranslationStep] = make_gtfn_translation,
    bindings: Callable[
        [GTFNConfig], workflow.Step[artifacts.ProgramSource, artifacts.ExtensionSource]
    ] = make_gtfn_bindings,
    compilation: Callable[
        [GTFNConfig], workflow.Step[artifacts.ExtensionSource, artifacts.CompilationArtifact]
    ] = make_gtfn_compiler,
) -> backend.Toolchain:
    """
    Build a GTFN toolchain.

    Args:
        cfg: The toolchain configuration. Defaults to `GTFNConfig()`.
        name_postfix: Appended to the toolchain name, which must stay unique.
        translation: Builder of the translation step, see
            `make_gtfn_compile_workflow`.
        bindings: Builder of the bindings step.
        compilation: Builder of the compilation step.

    Returns:
        The configured toolchain.

    Raises:
        ValueError: If a step builder returns a step configured for a device
            other than `cfg.device_type`.
    """
    if cfg is None:
        cfg = GTFNConfig()

    return backend.Toolchain(
        name=f"run_gtfn_{cfg.device_name}{name_postfix}",
        backend=make_gtfn_compile_workflow(
            cfg, translation=translation, bindings=bindings, compilation=compilation
        ),
        allocator=cfg.make_allocator(),
        frontend=backend.DEFAULT_TRANSFORMS,
    )


run_gtfn = make_gtfn_toolchain()

run_gtfn_gpu = make_gtfn_toolchain(GTFNConfig(gpu=True))

run_gtfn_no_transforms = make_gtfn_toolchain(
    name_postfix="_no_transforms",
    translation=functools.partial(make_gtfn_translation, enable_itir_transforms=False),
)
