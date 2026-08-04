# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import warnings
from collections.abc import Callable, MutableSequence, Sequence
from typing import Any, Final, TypeAlias

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


SDFG_ARG_EXTERNAL_WS_EVENT = gtx_wfdcommon.SDFG_ARG_EXTERNAL_WS_EVENT
SDFG_ARG_EXTERNAL_SYNC_STREAM = gtx_wfdcommon.SDFG_ARG_EXTERNAL_SYNC_STREAM


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

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_argtypes: list[dace.dtypes.Data]

    # The compiled program contains a callable object to update the SDFG arguments list.
    update_sdfg_ctype_arglist: Callable[
        [
            core_defs.DeviceType,
            Sequence[dace.dtypes.Data],
            Sequence[Any],
            MutableSequence[Any],
            common.OffsetProvider,
        ],
        None,
    ]

    # Processed argument vectors that are passed to `CompiledSDFG.fast_call()`. `None`
    #  means that it has not been initialized, i.e. no call was ever performed.
    #  - csdfg_argv: Arguments used for calling the actual compiled SDFG, will be updated.
    #  - csdfg_init_argv: Arguments used for initialization; used only the first time and
    #       never updated.
    csdfg_argv: MutableSequence[Any] | None
    csdfg_init_argv: Sequence[Any] | None
    external_workspace: gtx_wfdcommon.ExternalWorkspace | None = (
        None  # This attribute is set at runtime, before the first call.
    )
    external_sync_stream: Any | None = (
        None  # This attribute is set at runtime, before the first call.
    )
    _sync_event: Any | None = None

    def __init__(
        self,
        program: dace.CompiledSDFG,
        bind_func_name: str,
        binding_source_code: str,
    ):
        self.sdfg_program = program

        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_argtypes = list(program.sdfg.arglist().values())

        # The binding source code is Python tailored to this specific SDFG.
        # We dynamically compile that function and add it to the compiled program.
        global_namespace: dict[str, Any] = {}
        exec(binding_source_code, global_namespace)
        self.update_sdfg_ctype_arglist = global_namespace[bind_func_name]
        # For debug purpose, we set a unique module name on the compiled function.
        self.update_sdfg_ctype_arglist.__module__ = os.path.basename(program.sdfg.build_folder)

        # Since the SDFG hasn't been called yet.
        self.csdfg_argv = None
        self.csdfg_init_argv = None

    def _configure_external_workspace(self, **kwargs: Any) -> None:
        self.sdfg_program.initialize(**kwargs)
        if workspace_sizes := self.sdfg_program.get_workspace_sizes():
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
                self.sdfg_program.set_workspace(storage, workspace)

    def _add_stream_sync_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Add external workspace event/stream handles to SDFG kwargs if needed."""
        import cupy as cp

        self._sync_event = cp.cuda.Event()

        # Pass the raw pointer values as 64-bit integers. DaCe receives them as
        # `dace.uint64` scalar symbols and the tasklets cast them to the
        # appropriate CUDA/HIP handle types. When no external stream was
        # provided, the CUDA/HIP default stream is used.
        stream_ptr = cp.cuda.Stream(null=True).ptr if self.external_sync_stream is None else self.external_sync_stream.ptr
        kwargs = kwargs | {
            SDFG_ARG_EXTERNAL_WS_EVENT: self._sync_event.ptr,
            SDFG_ARG_EXTERNAL_SYNC_STREAM: stream_ptr,
        }
        return kwargs

    def construct_arguments(self, **kwargs: Any) -> None:
        """
        This function will process the arguments and store the processed argument
        vectors in `self.csdfg_args`, to call them use `self.fast_call()`.
        """
        if SDFG_ARG_EXTERNAL_WS_EVENT in self.sdfg_program.sdfg.symbols:
            kwargs = self._add_stream_sync_kwargs(kwargs)
        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg_argv, csdfg_init_argv = self.sdfg_program.construct_arguments(**kwargs)
            self._configure_external_workspace(**kwargs)
        # Note we only care about `csdfg_argv` (normal call), since we have to update it,
        #  we ensure that it is a `list`.
        self.csdfg_argv = [*csdfg_argv]
        self.csdfg_init_argv = csdfg_init_argv

    def fast_call(self) -> None:
        """
        Perform a call to the compiled SDFG using the previously generated argument
        vectors, see `self.construct_arguments()`.
        """
        assert self.csdfg_argv is not None and self.csdfg_init_argv is not None, (
            "Argument vector was not set properly."
        )
        self.sdfg_program.fast_call(
            self.csdfg_argv, self.csdfg_init_argv, do_gpu_check=config.DEBUG
        )

    def __call__(self, **kwargs: Any) -> None:
        """Call the compiled SDFG with the given arguments.

        Note that this function will not update the argument vectors stored inside
        `self`. Furthermore, it is not recommended to use this function as it is
        very slow.
        """
        warnings.warn(
            "Called an SDFG through the standard DaCe interface is not recommended, use `fast_call()` instead.",
            stacklevel=1,
        )
        result = self.sdfg_program(**kwargs)
        assert result is None


@dataclasses.dataclass(frozen=True)
class DaCeCompilationArtifact:
    """Result of a DaCe compilation: library path + SDFG bindings + the SDFG itself.

    The SDFG is carried inline as JSON because dace's load path
    (``get_program_handle``) needs an SDFG instance to wrap into the
    returned ``CompiledSDFG``, and the build folder may not contain a
    ``program.sdfg(z)`` dump under the upcoming minimal-build-dir mode.

    The SDFG we store here is the one on which we called `SDFG.compile(return_program_handle=False)`.
    Note that the `compile()` call has side effects, because it applies transformations
    to the SDFG, in order to enable code generation for the target platform.
    Since we pass `return_program_handle=False`, the `compile()` method does not
    return a `CompiledSDFG` instance, therefore we cannot access `CompiledSDFG.sdfg`,
    which would be the modified SDFG from which DaCe generates the C++/CUDA/HIP code.
    """

    library_path: pathlib.Path
    sdfg_json: str
    binding_source_code: str
    bind_func_name: str
    device_type: core_defs.DeviceType

    def load(self) -> stages.ExecutableProgram:
        # TODO(phimuell): Drop ``sdfg_json`` from the artifact once dace
        #   exposes a load path that doesn't require an SDFG instance to wrap
        #   into the returned ``CompiledSDFG``.
        sdfg = dace.SDFG.from_json(json.loads(self.sdfg_json))
        sdfg_program = dace_compiler.get_program_handle(self.library_path, sdfg)
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
    max_concurrent_gpu_streams: int = 0
    # we store the non-default values of `dace.Config` in order to include it in the stage fingerprint
    dace_config_nondefaults: dict[str, Any] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
            max_concurrent_gpu_streams=self.max_concurrent_gpu_streams,
        ):
            object.__setattr__(self, "dace_config_nondefaults", dace.Config._data.nondefaults())

    def __call__(self, inp: SDFGExtensionSource) -> DaCeCompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
            max_concurrent_gpu_streams=self.max_concurrent_gpu_streams,
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

            # ``build_folder_mode`` is set by ``dace_context``; resolve the library
            # path here so ``get_binary_name`` sees the same mode dace built under.
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
            library_path=library_path,
            sdfg_json=json.dumps(inp.program_source.source_code),
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
            device_type=self.device_type,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
