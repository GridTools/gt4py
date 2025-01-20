# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
import dataclasses
import functools
from typing import Any, Optional, Sequence

import dace
import factory
from dace.codegen.compiled_sdfg import _array_interface_ptr as get_array_interface_ptr

import gt4py.next.allocators as next_allocators
from gt4py._core import definitions as core_defs
from gt4py.eve.utils import content_hash
from gt4py.next import allocators as gtx_allocators, backend, common, config, utils as gtx_utils
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import arguments, languages, recipes, stages, step_types, workflow
from gt4py.next.otf.binding import interface
from gt4py.next.otf.compilation import cache
from gt4py.next.otf.languages import LanguageSettings
from gt4py.next.program_processors.runners.dace import (
    gtir_sdfg,
    sdfg_callable,
    sdfg_callable_args,
    transformations as gtx_transformations,
)


@dataclasses.dataclass(frozen=True)
class DaCeTranslator(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProgram, stages.ProgramSource[languages.SDFG, languages.LanguageSettings]
    ],
    step_types.TranslationStep[languages.SDFG, languages.LanguageSettings],
):
    device_type: core_defs.DeviceType
    auto_optimize: bool
    itir_transforms_off: bool = False

    def _language_settings(self) -> languages.LanguageSettings:
        return languages.LanguageSettings(
            formatter_key="", formatter_style="", file_extension="sdfg"
        )

    def generate_sdfg(
        self,
        ir: itir.Program,
        offset_provider: common.OffsetProvider,
        column_axis: Optional[common.Dimension],
        auto_opt: bool,
        on_gpu: bool,
    ) -> dace.SDFG:
        if not self.itir_transforms_off:
            ir = itir_transforms.apply_fieldview_transforms(ir, offset_provider=offset_provider)
        sdfg = gtir_sdfg.build_sdfg_from_gtir(
            ir, common.offset_provider_to_type(offset_provider), column_axis
        )

        if auto_opt:
            gtx_transformations.gt_auto_optimize(sdfg, gpu=on_gpu)
        elif on_gpu:
            # We run simplify to bring the SDFG into a canonical form that the gpu transformations
            # can handle. This is a workaround for an issue with scalar expressions that are
            # promoted to symbolic expressions and computed on the host (CPU), but the intermediate
            # result is written to a GPU global variable (https://github.com/spcl/dace/issues/1773).
            gtx_transformations.gt_simplify(sdfg)
            gtx_transformations.gt_gpu_transformation(sdfg, try_removing_trivial_maps=True)

        return sdfg

    def __call__(
        self, inp: stages.CompilableProgram
    ) -> stages.ProgramSource[languages.SDFG, LanguageSettings]:
        """Generate DaCe SDFG file from the GTIR definition."""
        program: itir.Program = inp.data
        assert isinstance(program, itir.Program)

        sdfg = self.generate_sdfg(
            program,
            inp.args.offset_provider,  # TODO(havogt): should be offset_provider_type once the transformation don't require run-time info
            inp.args.column_axis,
            auto_opt=self.auto_optimize,
            on_gpu=(self.device_type == gtx_allocators.CUPY_DEVICE),
        )

        param_types = tuple(
            interface.Parameter(param, arg_type)
            for param, arg_type in zip(sdfg.arg_names, inp.args.args)
        )

        module: stages.ProgramSource[languages.SDFG, languages.LanguageSettings] = (
            stages.ProgramSource(
                entry_point=interface.Function(program.id, param_types),
                source_code=sdfg.to_json(),
                library_deps=tuple(),
                language=languages.SDFG,
                language_settings=self._language_settings(),
                implicit_domain=inp.data.implicit_domain,
            )
        )
        return module


class DaCeTranslationStepFactory(factory.Factory):
    class Meta:
        model = DaCeTranslator


class CompiledDaceProgram(stages.ExtendedCompiledProgram):
    sdfg_program: dace.CompiledSDFG

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_arglist: list[tuple[str, dace.dtypes.Data]]

    def __init__(self, program: dace.CompiledSDFG, implicit_domain: bool):
        self.sdfg_program = program
        self.implicit_domain = implicit_domain
        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_arglist = [
            (arg_name, arg_type) for arg_name, arg_type in program.sdfg.arglist().items()
        ]

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        result = self.sdfg_program(*args, **kwargs)
        assert result is None

    def fast_call(self) -> None:
        result = self.sdfg_program.fast_call(*self.sdfg_program._lastargs)
        assert result is None


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
        CompiledDaceProgram,
    ],
    step_types.CompilationStep[languages.SDFG, languages.LanguageSettings, languages.Python],
):
    """Use the dace build system to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ) -> CompiledDaceProgram:
        sdfg = dace.SDFG.from_json(inp.program_source.source_code)

        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)
        sdfg.build_folder = src_dir / ".dacecache"

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "build_type", value=self.cmake_build_type.value)
            if self.device_type == core_defs.DeviceType.CPU:
                compiler_args = dace.config.Config.get("compiler", "cpu", "args")
                # disable finite-math-only in order to support isfinite/isinf/isnan builtins
                if "-ffast-math" in compiler_args:
                    compiler_args += " -fno-finite-math-only"
                if "-ffinite-math-only" in compiler_args:
                    compiler_args.replace("-ffinite-math-only", "")

                dace.config.Config.set("compiler", "cpu", "args", value=compiler_args)
            sdfg_program = sdfg.compile(validate=False)

        return CompiledDaceProgram(sdfg_program, inp.program_source.implicit_domain)


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler


def _convert_args(
    inp: CompiledDaceProgram,
    device: core_defs.DeviceType = core_defs.DeviceType.CPU,
    use_field_canonical_representation: bool = False,
) -> stages.CompiledProgram:
    sdfg_program = inp.sdfg_program
    sdfg = sdfg_program.sdfg
    on_gpu = True if device in [core_defs.DeviceType.CUDA, core_defs.DeviceType.ROCM] else False

    def decorated_program(
        *args: Any,
        offset_provider: common.OffsetProvider,
        out: Any = None,
    ) -> None:
        if out is not None:
            args = (*args, out)
        flat_args: Sequence[Any] = gtx_utils.flatten_nested_tuple(tuple(args))
        if inp.implicit_domain:
            # generate implicit domain size arguments only if necessary
            size_args = arguments.iter_size_args(args)
            flat_size_args: Sequence[int] = gtx_utils.flatten_nested_tuple(tuple(size_args))
            flat_args = (*flat_args, *flat_size_args)

        if sdfg_program._lastargs:
            kwargs = dict(zip(sdfg.arg_names, flat_args, strict=True))
            kwargs.update(sdfg_callable.get_sdfg_conn_args(sdfg, offset_provider, on_gpu))

            use_fast_call = True
            last_call_args = sdfg_program._lastargs[0]
            # The scalar arguments should be overridden with the new value; for field arguments,
            # the data pointer should remain the same otherwise fast_call cannot be used and
            # the arguments list has to be reconstructed.
            for i, (arg_name, arg_type) in enumerate(inp.sdfg_arglist):
                if isinstance(arg_type, dace.data.Array):
                    assert arg_name in kwargs, f"argument '{arg_name}' not found."
                    data_ptr = get_array_interface_ptr(kwargs[arg_name], arg_type.storage)
                    assert isinstance(last_call_args[i], ctypes.c_void_p)
                    if last_call_args[i].value != data_ptr:
                        use_fast_call = False
                        break
                else:
                    assert isinstance(arg_type, dace.data.Scalar)
                    assert isinstance(last_call_args[i], ctypes._SimpleCData)
                    if arg_name in kwargs:
                        # override the scalar value used in previous program call
                        actype = arg_type.dtype.as_ctypes()
                        last_call_args[i] = actype(kwargs[arg_name])
                    else:
                        # shape and strides of arrays are supposed not to change, and can therefore be omitted
                        assert sdfg_callable_args.is_field_symbol(
                            arg_name
                        ), f"argument '{arg_name}' not found."

            if use_fast_call:
                return inp.fast_call()

        sdfg_args = sdfg_callable.get_sdfg_args(
            sdfg,
            offset_provider,
            *flat_args,
            check_args=False,
            on_gpu=on_gpu,
            use_field_canonical_representation=use_field_canonical_representation,
        )

        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "allow_view_arguments", value=True)
            return inp(**sdfg_args)

    return decorated_program


def _no_bindings(inp: stages.ProgramSource) -> stages.CompilableSource:
    return stages.CompilableSource(program_source=inp, binding_source=None)


class DaCeWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(
            lambda: config.CMAKE_BUILD_TYPE
        )
        auto_optimize: bool = False

    translation = factory.SubFactory(
        DaCeTranslationStepFactory,
        device_type=factory.SelfAttribute("..device_type"),
        auto_optimize=factory.SelfAttribute("..auto_optimize"),
    )
    bindings = _no_bindings
    compilation = factory.SubFactory(
        DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            _convert_args,
            device=o.device_type,
        )
    )


def compilation_hash(otf_closure: stages.CompilableProgram) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile.

    RThis function was copied from 'next.program_processors.runners.gtfn'
    """
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


class DaCeFieldviewBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        name_device = "cpu"
        name_cached = ""
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
            DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            auto_optimize=factory.SelfAttribute("..auto_optimize"),
        )
        auto_optimize = factory.Trait(name_postfix="_opt")

    name = factory.LazyAttribute(
        lambda o: f"run_dace_{o.name_device}{o.name_cached}{o.name_postfix}"
    )

    executor = factory.LazyAttribute(lambda o: o.otf_workflow)
    allocator = next_allocators.StandardCPUFieldBufferAllocator()
    transforms = backend.DEFAULT_TRANSFORMS


run_dace_cpu = DaCeFieldviewBackendFactory(cached=True, auto_optimize=True)
run_dace_cpu_noopt = DaCeFieldviewBackendFactory(cached=True, auto_optimize=False)

run_dace_gpu = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_noopt = DaCeFieldviewBackendFactory(gpu=True, cached=True, auto_optimize=False)
