# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import os
import pathlib
from collections.abc import Callable
from typing import Any, Final, Sequence, TypeAlias

import dace
import dace.codegen.compiler as dace_compiler
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.eve import extended_typing as xtyping
from gt4py.next import common, config, fingerprinting
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import (
    common as gtx_wfdcommon,
    decoration as gtx_wfddecoration,
)


_COMPILE_COMPLETE_MARKER: Final = ".gt4py_compile_complete"


SDFGExtensionSource: TypeAlias = stages.ExtensionSource[
    code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec
]


def _add_tx_markers(program_source: SDFGExtensionSource) -> tuple[SDFGExtensionSource, dace.SDFG]:
    """
    Add GPU TX markers to the SDFG of the given extension source.

    Returns the modified extension source and the modified SDFG.
    """
    sdfg = dace.SDFG.from_json(program_source.program_source.source_code)

    has_gpu_schedule = any(
        getattr(node, "schedule", dace.dtypes.ScheduleType.Default) in dace.dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )

    if not has_gpu_schedule:
        return program_source, sdfg  # No GPU schedule, no need to add TX markers.

    sdfg.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS
    for node, _ in sdfg.all_nodes_recursive():
        # Also adds markers to map scopes that are NOT scheduled on GPU
        if isinstance(node, (dace.nodes.MapEntry, dace.sdfg.SDFGState)):
            node.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS

    sdfg_json = gtx_wfdcommon.serialize_sdfg_as_json(sdfg)

    new_program_source = dataclasses.replace(
        program_source,
        program_source=dataclasses.replace(program_source.program_source, source_code=sdfg_json),  # type: ignore[arg-type] # The source code is typed as a `str`, but we assign a JSON dictionary.
    )

    return new_program_source, sdfg


def _map_storage_to_device(storage: dace.StorageType) -> core_defs.DeviceType:
    if storage == dace.StorageType.CPU_Heap:
        device = core_defs.DeviceType.CPU
    elif storage == dace.StorageType.GPU_Global:
        if core_defs.CUPY_DEVICE_TYPE is None:
            raise ValueError(
                f"Can not map storage type '{storage}' to a device: no GPU device"
                " type is configured ('core_defs.CUPY_DEVICE_TYPE' is None)."
            )
        device = core_defs.CUPY_DEVICE_TYPE
    else:
        raise ValueError(f"Unsupported storage type '{storage}' for external workspace allocation.")

    return device


def _validate_external_workspace(
    wsp: xtyping.ArrayInterface | xtyping.CUDAArrayInterface, storage: dace.StorageType, nbytes: int
) -> None:
    """Validate that the provided ``wsp`` workspace satisfies the requirements.

    Args:
        wsp: The external workspace to check.
        storage: SDFG storage type the workspace buffer is being installed for.
        nbytes: Size in bytes required.

    Raises:
        TypeError: If ``wsp`` exposes neither ``__array_interface__`` nor
            ``__cuda_array_interface__``.
        ValueError: If ``wsp`` exposes ``nbytes`` and it is smaller than the
            required ``nbytes``. Buffers that do not expose ``nbytes`` are
            accepted on a trust basis (their size can not be checked here).
    """
    if storage == dace.StorageType.GPU_Global:
        if not xtyping.supports_cuda_array_interface(wsp):
            raise TypeError(
                f"External workspace for storage {storage!r} must expose `__cuda_array_interface__` (got {type(wsp).__name__!r})."
            )
    elif storage == dace.StorageType.CPU_Heap:
        if not xtyping.supports_array_interface(wsp):
            raise TypeError(
                f"External workspace for storage {storage!r} must expose `__array_interface__` (got {type(wsp).__name__!r})."
            )
    else:
        raise ValueError(f"Unsupported storage type {storage!r} for external workspace allocation.")

    if (wsp_nbytes := getattr(wsp, "nbytes", None)) is not None and wsp_nbytes < nbytes:
        raise ValueError(
            f"External workspace buffer is {wsp_nbytes} bytes for storage {storage!r}, but at least {nbytes} bytes were required."
        )


class CompiledDaceProgram:
    sdfg_program: dace.CompiledSDFG

    # Callable to process the GT4Py arguments and offset providers to bring them in a form suitable for calling.
    argument_preprocessing_function: Callable[
        [Sequence[Any], common.OffsetProvider, int, Any], tuple[Any, ...]
    ]

    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = (
        None  # This attribute is set at runtime, before the first call.
    )

    # Whether the SDFG has transients with `AllocationLifetime.External`, i.e. whether
    # the caller has to install a workspace before the program can run. Determined
    # once at load time so that the (non-trivial) workspace configuration is only
    # performed for the programs that actually need it.
    requires_external_workspace: bool

    def __init__(
        self,
        program: dace.CompiledSDFG,
        bind_func_name: str,
        binding_source_code: str,
    ):
        self.sdfg_program = program
        self.requires_external_workspace = any(
            desc.lifetime == dace.dtypes.AllocationLifetime.External
            for _, _, desc in program.sdfg.arrays_recursive()
        )

        # The binding source code is Python tailored to this specific SDFG.
        # We dynamically compile that function and add it to the compiled program.
        global_namespace: dict[str, Any] = {}
        exec(binding_source_code, global_namespace)

        self.argument_preprocessing_function = global_namespace[bind_func_name]
        # For debug purpose, we set a unique module name on the compiled function.
        self.argument_preprocessing_function.__module__ = os.path.basename(
            program.sdfg.build_folder
        )

    def configure_external_workspace(self, **kwargs: Any) -> None:
        """Install the caller-provided workspace buffers on the compiled SDFG.

        This eagerly initializes the SDFG state, queries the required workspace
        size per storage type and hands the matching buffer of
        `self.external_workspace` to DaCe. It has to run before the first call,
        because an SDFG with external transients refuses to run with no
        workspace installed.

        The required sizes depend on the symbol values passed here, so they are
        only valid for calls made with the same symbols. This is called once,
        with the arguments of the first call; keeping the workspace large enough
        for subsequent calls is the caller's responsibility.

        Args:
            kwargs: The SDFG call arguments. Only the free symbols among them are
                used; the remaining entries are ignored by DaCe.

        Raises:
            RuntimeError: If the SDFG needs a workspace but none was set for the
                required device.
            TypeError: If a workspace buffer exposes no suitable array interface.
            ValueError: If a workspace buffer is too small, or is required for an
                unsupported storage type.
        """
        self.sdfg_program.initialize(**kwargs)
        if workspace_sizes := self.sdfg_program.get_workspace_sizes(**kwargs):
            if self.external_workspace is None:
                raise RuntimeError(
                    "External workspace is not set. Please call `set_external_workspace()`"
                    " before the first call to the program."
                )
            for storage, required_nbytes in workspace_sizes.items():
                device = _map_storage_to_device(storage)
                workspace = self.external_workspace.get(device)
                if workspace is None:
                    raise RuntimeError(f"External workspace for device {device} not found.")
                _validate_external_workspace(workspace, storage, required_nbytes)
                self.sdfg_program.set_workspace(storage, workspace, **kwargs)

    def __call__(self, **kwargs: Any) -> None:
        """Call the compiled SDFG with the given arguments.

        A `CompiledDaceProgram` should not be called directly. Instead
        `gt4py.next.program_processors.runners.dace.workflow.decoration.convert_args()`
        should be used to obtain a callable.
        """
        raise NotImplementedError(
            "A `CompiledDaceProgram` can not be called directly. Instead use "
            "`gt4py.next.program_processors.runners.dace.workflow.decoration.convert_args()`."
        )


@dataclasses.dataclass(frozen=True)
class DaCeCompilationArtifact:
    """Result of a DaCe compilation: library path + SDFG bindings + the SDFG itself."""

    sdfg_build_folder: pathlib.Path
    binding_source_code: str
    bind_func_name: str
    device_type: core_defs.DeviceType

    def load(self) -> stages.ExecutableProgram:
        sdfg_program = dace_compiler.load_precompiled_sdfg(self.sdfg_build_folder, sdfg=None)
        sdfg_program.gpu_error_check = False  # Not useful in asynchronous launches.
        program = CompiledDaceProgram(sdfg_program, self.bind_func_name, self.binding_source_code)
        return gtx_wfddecoration.DaCeDecoratedProgram(program, device_type=self.device_type)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeCompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.ExtensionSource[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        DaCeCompilationArtifact,
    ],
    definitions.CompilationStep[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
):
    """Run the DaCe build system and produce an on-disk ``DaCeCompilationArtifact``."""

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    add_gpu_trace_markers: bool = dataclasses.field(
        default_factory=lambda: config.ADD_GPU_TRACE_MARKERS
    )
    cmake_build_type: config.CMakeBuildType = dataclasses.field(
        default_factory=lambda: config.CMAKE_BUILD_TYPE
    )
    # We store the non-default values of `dace.Config` in order to include it in the stage fingerprint
    # NOTE: They do not include the non default keys set through DaCe environment variables.
    dace_config_nondefaults: dict[str, Any] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            object.__setattr__(self, "dace_config_nondefaults", dace.Config._data.nondefaults())

    def __call__(self, inp: SDFGExtensionSource) -> DaCeCompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            # Add TX markers to the generated GPU code for trace visualization tools.
            if self.add_gpu_trace_markers and self.device_type == core_defs.CUPY_DEVICE_TYPE:
                inp, sdfg = _add_tx_markers(inp)
            else:
                sdfg = dace.SDFG.from_json(inp.program_source.source_code)

            # Fingerprint the non-default ``dace.Config`` so the SDFG rebuilds when the
            # user changes the backend configuration (PR #2650).
            sdfg_build_folder = gtx_cache.get_cache_folder(
                inp,
                self.cache_lifetime,
                build_context_id=fingerprinting.strict_fingerprinter(self.dace_config_nondefaults),
            )
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)

            # Configure the SDFG build folder
            sdfg.build_folder = sdfg_build_folder

            # `compiler.build_folder_mode` is set by `dace_context()`; resolve the library
            #  path here so `get_binary_name()` sees the same mode DaCe built under.
            library_path = dace_compiler.get_binary_name(
                object_folder=sdfg_build_folder, sdfg_name=sdfg.name
            )

            with locking.lock(sdfg_build_folder):
                # With `compiler.use_cache=True` dace reuses a cached library on mere
                # *existence*, without validating it; an interrupted build can leave a
                # truncated, unloadable library behind. The marker is written only
                # after a completed compile: no marker -> drop the stale library so
                # dace rebuilds it instead of handing it out.
                marker = sdfg_build_folder / _COMPILE_COMPLETE_MARKER
                if not marker.exists():
                    for stale in (
                        library_path,
                        *sdfg_build_folder.glob(f"libdacestub_{sdfg.name}.*"),
                    ):
                        stale.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                sdfg.compile(validate=False, return_program_handle=False)
                marker.touch()

        assert inp.binding_source is not None
        return DaCeCompilationArtifact(
            sdfg_build_folder=sdfg_build_folder,
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
            device_type=self.device_type,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
