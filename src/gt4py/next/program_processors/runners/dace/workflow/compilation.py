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
import pickle
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
from gt4py.next.program_processors.runners.dace.transformations.auto_optimize import (
    AllocationRequest,
    ExternalMemoryAllocator,
    ExternalWorkspace,
)
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


def workspace_storage_to_device_mapping(storage: dace.StorageType) -> core_defs.DeviceType:
    if storage == dace.StorageType.CPU_Heap:
        device = core_defs.DeviceType.CPU
    elif storage == dace.StorageType.GPU_Global:
        assert core_defs.CUPY_DEVICE_TYPE is not None
        device = core_defs.CUPY_DEVICE_TYPE
    else:
        raise ValueError(f"Unsupported storage type '{storage}' for external workspace allocation.")

    return device


def _validate_external_workspace(
    storage: dace.StorageType, request: AllocationRequest, wsp: ExternalWorkspace
) -> None:
    """Validate that ``wsp`` satisfies ``request`` for ``storage``.

    Args:
        storage: SDFG storage type the workspace buffer is being installed for.
        request: Allocation request that was issued.
        wsp: External workspace returned by the external allocator.

    Raises:
        TypeError: If ``wsp`` exposes neither ``__array_interface__`` nor
            ``__cuda_array_interface__``.
        ValueError: If ``wsp`` is smaller than ``request.nbytes`` or its
            base pointer is not aligned to ``request.alignment`` bytes.
    """
    if not (xtyping.supports_array_interface(wsp) or xtyping.supports_cuda_array_interface(wsp)):
        raise TypeError(
            f"External memory allocator returned {type(wsp).__name__!r} for storage "
            f"{storage!r}, which does not expose `__array_interface__` or "
            f"`__cuda_array_interface__`."
        )
    nbytes = getattr(wsp, "nbytes", None)
    if nbytes is not None and nbytes < request.nbytes:
        raise ValueError(
            f"External memory allocator returned a buffer of {nbytes} bytes for storage "
            f"{storage!r}, but at least {request.nbytes} were required."
        )
    # Validate alignment against the base pointer DaCe will hand to the SDFG
    # (see ``dace.dtypes.array_interface_ptr``). The ``data`` field is
    # optional on the host array interface; if it is missing the alignment
    # contract is trust-based and the check is skipped, mirroring ``nbytes``.
    interface = (
        getattr(wsp, "__cuda_array_interface__", None)
        if storage == dace.StorageType.GPU_Global
        else getattr(wsp, "__array_interface__", None)
    )
    data = interface.get("data") if interface is not None else None
    if data is not None and request.alignment > 1 and data[0] % request.alignment != 0:
        raise ValueError(
            f"External memory allocator returned a buffer for storage {storage!r} "
            f"whose base pointer ({data[0]}) is not aligned to the required "
            f"{request.alignment} bytes."
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
    external_memory_allocator: ExternalMemoryAllocator | None
    external_workspaces: dict[dace.StorageType, ExternalWorkspace]

    def __init__(
        self,
        program: dace.CompiledSDFG,
        bind_func_name: str,
        binding_source_code: str,
        external_memory_allocator: ExternalMemoryAllocator | None = None,
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
        self.external_memory_allocator = external_memory_allocator
        self.external_workspaces = {}

    def _configure_external_workspaces(self, **kwargs: Any) -> None:
        if self.external_workspaces:
            # We already allocated the external workspaces, no need to do it again.
            return

        # DaCe computes workspace sizes during ``initialize`` and stores them
        # for subsequent ``get_workspace_sizes``/``set_workspace`` calls.
        self.sdfg_program.initialize(**kwargs)
        if workspace_sizes := self.sdfg_program.get_workspace_sizes():
            if self.external_memory_allocator is None:
                raise ValueError(
                    "SDFG requires external workspaces, but no allocator was provided."
                )
            for storage, required_nbytes in workspace_sizes.items():
                device = workspace_storage_to_device_mapping(storage)
                request = AllocationRequest(nbytes=required_nbytes, device=device)
                workspace = self.external_memory_allocator.allocate(request)
                _validate_external_workspace(storage, request, workspace)
                self.sdfg_program.set_workspace(storage, workspace)
                # Keep the workspace buffers alive as long as the compiled program lives.
                self.external_workspaces[storage] = workspace

    def finalize(self) -> None:
        """Release external workspaces.

        Finalizes the underlying ``sdfg_program`` and calls ``deallocate``
        once per allocated storage type. Safe to call multiple times: after
        the first call the per-storage workspace buffers are dropped from
        ``external_workspaces`` and subsequent calls are no-ops. A ``None``
        allocator performs no work but still clears any externally-installed
        workspaces.

        Failures during deallocation are surfaced as warnings rather than
        raised, so that one failing buffer does not prevent the remaining
        workspaces from being released.
        """
        if self.external_memory_allocator is not None:
            for wsp in self.external_workspaces.values():
                try:
                    self.external_memory_allocator.deallocate(wsp)
                except Exception:
                    warnings.warn(
                        f"Failed to deallocate external workspace "
                        f"({type(wsp).__name__!r}); it may be leaked.",
                        stacklevel=1,
                    )
        self.external_workspaces = {}

    def construct_arguments(self, **kwargs: Any) -> None:
        """
        This function will process the arguments and store the processed argument
        vectors in `self.csdfg_args`, to call them use `self.fast_call()`.
        """
        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg_argv, csdfg_init_argv = self.sdfg_program.construct_arguments(**kwargs)
            self._configure_external_workspaces(**kwargs)
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


class AllocatorNotPicklableError(TypeError):
    """Raised when an ``external_memory_allocator`` cannot be pickled.

    The allocator is part of the compilation artifact and is pickled when
    compilation is offloaded to a worker process. Allocators that can not be
    pickled -- typically closures, lambdas, or classes defined inside a
    function -- would otherwise degrade silently to in-process compilation
    via a generic runner warning. This error surfaces the contract failure
    early, at backend construction, with the original :mod:`pickle` error
    chained as ``__cause__``.
    """


def _check_allocator_picklable(allocator: ExternalMemoryAllocator) -> None:
    """Fail fast if ``allocator`` is not picklable.

    Args:
        allocator: The allocator to probe; must not be ``None``.

    Raises:
        AllocatorNotPicklableError: If ``pickle.dumps(allocator)`` raises.
    """
    try:
        pickle.dumps(allocator)
    except Exception as error:  # pickle raises arbitrary exceptions
        raise AllocatorNotPicklableError(
            f"external_memory_allocator {allocator!r} is not picklable: {error!r}."
            " The allocator is part of the compilation artifact and is pickled"
            " when compilation is offloaded to a worker process. Use a"
            " module-level class or functools.partial of picklable callables."
        ) from error


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
    external_memory_allocator: ExternalMemoryAllocator | None = None

    def load(self) -> stages.ExecutableProgram:
        # TODO(phimuell): Drop ``sdfg_json`` from the artifact once dace
        #   exposes a load path that doesn't require an SDFG instance to wrap
        #   into the returned ``CompiledSDFG``.
        sdfg = dace.SDFG.from_json(json.loads(self.sdfg_json))
        sdfg_program = dace_compiler.get_program_handle(self.library_path, sdfg)
        program = CompiledDaceProgram(
            sdfg_program,
            self.bind_func_name,
            self.binding_source_code,
            external_memory_allocator=self.external_memory_allocator,
        )
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
    #: Allocator providing external workspace memory when
    #: ``transient_memory_mode`` is ``EXTERNAL``. Must be picklable (a
    #: module-level class or :py:func:`functools.partial` of picklable
    #: callables is recommended); probed at construction time.
    external_memory_allocator: ExternalMemoryAllocator | None = None
    add_gpu_trace_markers: bool = dataclasses.field(
        default_factory=lambda: config.ADD_GPU_TRACE_MARKERS
    )
    cmake_build_type: config.CMakeBuildType = dataclasses.field(
        default_factory=lambda: config.CMAKE_BUILD_TYPE
    )
    # we store the non-default values of `dace.Config` in order to include it in the stage fingerprint
    dace_config_nondefaults: dict[str, Any] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # The allocator is part of the compilation artifact and is pickled
        # when compilation is offloaded to a worker process. Probe it here,
        # at backend construction, so a non-picklable allocator (closure,
        # lambda, local class) fails fast with an actionable error instead
        # of silently degrading to in-process compilation.
        if self.external_memory_allocator is not None:
            _check_allocator_picklable(self.external_memory_allocator)
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
            external_memory_allocator=self.external_memory_allocator,
        )


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
