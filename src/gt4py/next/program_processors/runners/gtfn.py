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
from typing import Any, Optional

import diskcache
import factory
import numpy as np
from flufl import lock

import gt4py._core.definitions as core_defs
import gt4py.next.allocators as next_allocators
from gt4py.next import backend, common, config, field_utils
from gt4py.next.embedded import nd_array_field
from gt4py.next.otf import arguments, recipes, stages, workflow
from gt4py.next.otf.binding import nanobind
from gt4py.next.otf.compilation import compiler
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
    inp: stages.ExtendedCompiledProgram, device: core_defs.DeviceType = core_defs.DeviceType.CPU
) -> stages.CompiledProgram:
    def decorated_program(
        *args: Any,
        offset_provider: dict[str, common.Connectivity | common.Dimension],
        out: Any = None,
    ) -> None:
        # Note: this function is on the hot path and needs to have minimal overhead.
        if out is not None:
            args = (*args, out)
        converted_args = (convert_arg(arg) for arg in args)
        conn_args = extract_connectivity_args(offset_provider, device)
        # generate implicit domain size arguments only if necessary, using `iter_size_args()`
        return inp(
            *converted_args,
            *(arguments.iter_size_args(args) if inp.implicit_domain else ()),
            *conn_args,
        )

    return decorated_program


def extract_connectivity_args(
    offset_provider: dict[str, common.Connectivity | common.Dimension], device: core_defs.DeviceType
) -> list[tuple[core_defs.NDArrayObject, tuple[int, ...]]]:
    # Note: this function is on the hot path and needs to have minimal overhead.
    args: list[tuple[core_defs.NDArrayObject, tuple[int, ...]]] = []
    # Note: the order here needs to agree with the order of the generated bindings
    for conn in offset_provider.values():
        if (ndarray := getattr(conn, "ndarray", None)) is not None:
            assert common.is_neighbor_table(conn)
            assert field_utils.verify_device_field_type(conn, device)
            args.append((ndarray, (0, 0)))
            continue
        assert isinstance(conn, common.Dimension)
    return args


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

        lock_dir.mkdir(parents=True, exist_ok=True)
        lockfile = str(lock_dir / "file_cache.lock")
        with lock.Lock(lockfile, lifetime=10):  # type: ignore[attr-defined] # mypy not smart enough to understand custom export logic
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
                    hash_function=stages.fingerprint_compilable_program,
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
            device_type=core_defs.CUPY_DEVICE_TYPE or core_defs.DeviceType.CUDA,
            name_device="gpu",
        )
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        device_type = core_defs.DeviceType.CPU
        hash_function = stages.compilation_hash
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
