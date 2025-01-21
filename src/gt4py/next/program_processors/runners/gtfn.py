# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import pathlib
import tempfile
import warnings
from typing import Any, Optional

import diskcache
import factory
import filelock

import gt4py._core.definitions as core_defs
import gt4py.next.allocators as next_allocators
from gt4py.eve import utils
from gt4py.eve.utils import content_hash
from gt4py.next import backend, common, config
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
from gt4py.next.otf.compilation.build_systems import compiledb
from gt4py.next.program_processors.codegens.gtfn import gtfn_module


# TODO(ricoh): Add support for the whole range of arguments that can be passed to a fencil.
def convert_arg(arg: Any) -> Any:
    if isinstance(arg, tuple):
        return tuple(convert_arg(a) for a in arg)
    if isinstance(arg, common.Field):
        arr = arg.ndarray
        origin = getattr(arg, "__gt_origin__", tuple([0] * len(arg.domain)))
        return arr, origin
    else:
        return arg


def convert_args(
    inp: stages.ExtendedCompiledProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.CompiledProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.Connectivity | common.Dimension],
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)
        converted_args = [convert_arg(arg) for arg in args]
        conn_args = extract_connectivity_args(offset_provider, device)
        # generate implicit domain size arguments only if necessary, using `iter_size_args()`
        return inp(
            *converted_args,
            *(arguments.iter_size_args(args) if inp.implicit_domain else ()),
            *conn_args,
        )

    return decorated_program


def _ensure_is_on_device(
    connectivity_arg: core_defs.NDArrayObject, device: core_defs.DeviceType
) -> core_defs.NDArrayObject:
    if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM]:
        import cupy as cp

        if not isinstance(connectivity_arg, cp.ndarray):
            warnings.warn(
                "Copying connectivity to device. For performance make sure connectivity is provided on device.",
                stacklevel=2,
            )
            return cp.asarray(connectivity_arg)
    return connectivity_arg


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension], device: core_defs.DeviceType
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # note: the order here needs to agree with the order of the generated bindings
    args: list[tuple[core_defs.NDArrayObject, tuple[int, ...]]] = []
    for name, conn in offset_provider.items():
        if isinstance(conn, common.Connectivity):
            if not common.is_neighbor_table(conn):
                raise NotImplementedError(
                    "Only 'NeighborTable' connectivities implemented at this point."
                )
            # copying to device here is a fallback for easy testing and might be removed later
            conn_arg = _ensure_is_on_device(conn.ndarray, device)
            args.append((conn_arg, tuple([0] * 2)))
        elif isinstance(conn, common.Dimension):
            pass
        else:
            raise AssertionError(
                f"Expected offset provider '{name}' to be a 'Connectivity' or 'Dimension', "
                f"but got '{type(conn).__name__}'."
            )
    return args


def compilation_hash(otf_closure: stages.CompilableProgram) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile."""
    offset_provider = otf_closure.args.offset_provider
    return hash(
        (
            otf_closure.data,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            content_hash(tuple(arg for arg in otf_closure.args.args)),
            # Directly using the `id` of the offset provider is not possible as the decorator adds
            # the implicitly defined ones (i.e. to allow the `TDim + 1` syntax) resulting in a
            # different `id` every time. Instead use the `id` of each individual offset provider.
            tuple((k, id(v)) for (k, v) in offset_provider.items()) if offset_provider else None,
            otf_closure.args.column_axis,
        )
    )


def fingerprint_compilable_program(inp: stages.CompilableProgram) -> str:
    """
    Generates a unique hash string for a stencil source program representing
    the program, sorted offset_provider, and column_axis.
    """
    program: itir.Program = inp.data
    offset_provider: common.OffsetProvider = inp.args.offset_provider
    column_axis: Optional[common.Dimension] = inp.args.column_axis

    program_hash = utils.content_hash(
        (
            program,
            sorted(offset_provider.items(), key=lambda el: el[0]),
            column_axis,
        )
    )

    return program_hash


class FileCache(diskcache.Cache):
    """
    This class extends `diskcache.Cache` to ensure the cache is properly
    - opened when accessed by multiple processes using a file lock. This guards the creating of the
    cache object, which has been reported to cause `sqlite3.OperationalError: database is locked`
    errors and slow startup times when multiple processes access the cache concurrently. While this
    issue occurred frequently and was observed to be fixed on distributed file systems, the lock
    does not guarantee correct behavior in particular for accesses to the cache (beyond opening)
    since the underlying SQLite database is unreliable when stored on an NFS based file system.
    It does however ensure correctness of concurrent cache accesses on a local file system. See
    #1745 for more details.
    - closed upon deletion, i.e. it ensures that any resources associated with the cache are
    properly released when the instance is garbage collected.
    """

    def __init__(self, directory: Optional[str | pathlib.Path] = None, **settings: Any) -> None:
        if directory:
            lock_dir = pathlib.Path(directory).parent
        else:
            lock_dir = pathlib.Path(tempfile.gettempdir())

        lock = filelock.FileLock(lock_dir / "file_cache.lock")
        with lock:
            super().__init__(directory=directory, **settings)

        self._init_complete = True

    def __del__(self) -> None:
        if getattr(self, "_init_complete", False):  # skip if `__init__` didn't finished
            self.close()


class GTFNCompileWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(
            lambda: config.CMAKE_BUILD_TYPE
        )
        builder_factory: compiler.BuildSystemProjectGenerator = factory.LazyAttribute(
            lambda o: compiledb.CompiledbFactory(cmake_build_type=o.cmake_build_type)
        )

        cached_translation = factory.Trait(
            translation=factory.LazyAttribute(
                lambda o: workflow.CachedStep(
                    o.bare_translation,
                    hash_function=fingerprint_compilable_program,
                    cache=FileCache(str(config.BUILD_CACHE_DIR / "gtfn_cache")),
                )
            ),
        )

        bare_translation = factory.SubFactory(
            gtfn_module.GTFNTranslationStepFactory,
            device_type=factory.SelfAttribute("..device_type"),
        )

    translation = factory.LazyAttribute(lambda o: o.bare_translation)

    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableSource] = (
        nanobind.bind_source
    )
    compilation = factory.SubFactory(
        compiler.CompilerFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        builder_factory=factory.SelfAttribute("..builder_factory"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(convert_args, device=o.device_type)
    )


class GTFNBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
        name_temps = ""
        name_postfix = ""
        gpu = factory.Trait(
            allocator=next_allocators.StandardGPUFieldBufferAllocator(),
            device_type=next_allocators.CUPY_DEVICE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        device_type = core_defs.DeviceType.CPU
        hash_function = compilation_hash
        otf_workflow = factory.SubFactory(
            GTFNCompileWorkflowFactory, device_type=factory.SelfAttribute("..device_type")
        )

    name = factory.LazyAttribute(
        lambda o: f"run_gtfn_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


run_gtfn = GTFNBackendFactory()

run_gtfn_imperative = GTFNBackendFactory(
    name_postfix="_imperative", otf_workflow__translation__use_imperative_backend=True
)

run_gtfn_cached = GTFNBackendFactory(cached=True, otf_workflow__cached_translation=True)

run_gtfn_gpu = GTFNBackendFactory(gpu=True)

run_gtfn_gpu_cached = GTFNBackendFactory(gpu=True, cached=True)

run_gtfn_no_transforms = GTFNBackendFactory(
    otf_workflow__bare_translation__enable_itir_transforms=False
)
