# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import abc
import os
import textwrap
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from gt4py import backend as gt_backend
from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.backend.module_generator import BaseModuleGenerator, ModuleData
from gt4py.definitions import AccessKind
from gtc import gtir
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_pipeline import OirPipeline


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder
    from gt4py.stencil_object import StencilObject
    from gt4py.storage.storage import Storage


def iir_is_not_empty(implementation_ir: gt_ir.StencilImplementation) -> bool:
    return bool(implementation_ir.multi_stages)


def gtir_is_not_empty(pipeline: GtirPipeline) -> bool:
    node = pipeline.full()
    return bool(node.iter_tree().if_isinstance(gtir.ParAssignStmt).to_list())


def iir_has_effect(implementation_ir: gt_ir.StencilImplementation) -> bool:
    return bool(implementation_ir.has_effect)


def gtir_has_effect(pipeline: GtirPipeline) -> bool:
    return True


class PyExtModuleGenerator(BaseModuleGenerator):
    """
    Module Generator for use with backends that generate c++ python extensions.

    Will either use ImplementationIR or GTIR depending on the backend's USE_LEGACY_TOOLCHAIN
    class attribute. Using with other IRs requires subclassing and overriding ``_is_not_empty()``
    and ``_has_effect()`` methods.
    """

    pyext_module_name: Optional[str]
    pyext_file_path: Optional[str]

    def __init__(self):
        super().__init__()
        self.pyext_module_name = None
        self.pyext_file_path = None

    def __call__(
        self,
        args_data: ModuleData,
        builder: Optional["StencilBuilder"] = None,
        **kwargs: Any,
    ) -> str:
        self.pyext_module_name = kwargs["pyext_module_name"]
        self.pyext_file_path = kwargs["pyext_file_path"]
        return super().__call__(args_data, builder, **kwargs)

    def _is_not_empty(self) -> bool:
        if self.pyext_module_name is None:
            return False
        if self.builder.backend.USE_LEGACY_TOOLCHAIN:
            return iir_is_not_empty(self.builder.implementation_ir)
        return gtir_is_not_empty(self.builder.gtir_pipeline)

    def generate_imports(self) -> str:
        source = ["from gt4py import utils as gt_utils"]
        if self._is_not_empty():
            assert self.pyext_file_path is not None
            file_path = 'f"{{pathlib.Path(__file__).parent.resolve()}}/{}"'.format(
                os.path.basename(self.pyext_file_path)
            )
            source.append(
                textwrap.dedent(
                    f"""
                pyext_module = gt_utils.make_module_from_file(
                    "{self.pyext_module_name}", {file_path}, public_import=True
                )
                """
                )
            )
        return "\n".join(source)

    def _has_effect(self) -> bool:
        if not self._is_not_empty():
            return False
        if self.builder.backend.USE_LEGACY_TOOLCHAIN:
            return iir_has_effect(self.builder.implementation_ir)
        return gtir_has_effect(self.builder.gtir_pipeline)

    def generate_implementation(self) -> str:
        ir = self.builder.gtir
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        params_decls = {decl.name: decl for decl in ir.params}
        args: List[str] = []
        for arg in ir.api_signature:
            if arg.name not in self.args_data.unreferenced:
                args.append(arg.name)
                if isinstance(params_decls.get(arg.name, None), gtir.FieldDecl):
                    args.append("list(_origin_['{}'])".format(arg.name))

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self._has_effect():
            source = textwrap.dedent(
                f"""
                # Load or generate a GTComputation object for the current domain size
                pyext_module.run_computation({",".join(["list(_domain_)", *args, "exec_info"])})
                """
            )
            sources.extend(source.splitlines())
        else:
            sources.extend("\n")

        return sources.text


class GeneratorClass:
    TEMPLATE_FILES: Dict[str, str]

    @abc.abstractmethod
    def __init__(self, class_name: str, module_name: str, backend: str):
        pass

    @abc.abstractmethod
    def __call__(self, ir: gtir.Stencil) -> Dict[str, Dict[str, str]]:
        """Return a dict with the keys 'computation' and 'bindings' to dicts of filenames to source."""
        pass


class BaseGTBackend(gt_backend.BasePyExtBackend, gt_backend.CLIBackendMixin):

    GT_BACKEND_OPTS = {
        "add_profile_info": {"versioning": True, "type": bool},
        "clean": {"versioning": False, "type": bool},
        "debug_mode": {"versioning": True, "type": bool},
        "verbose": {"versioning": False, "type": bool},
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
    }

    GT_BACKEND_T: str

    MODULE_GENERATOR_CLASS = PyExtModuleGenerator

    PYEXT_GENERATOR_CLASS: Type[GeneratorClass]

    @abc.abstractmethod
    def generate(self) -> Type["StencilObject"]:
        pass

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(ir=self.builder.gtir)
        return {dir_name: src_files["computation"]}

    def generate_bindings(
        self, language_name: str, *, ir: Any = None
    ) -> Dict[str, Union[str, Dict]]:
        if not ir:
            ir = self.builder.gtir
        if language_name != "python":
            return super().generate_bindings(language_name)
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(ir=ir)
        return {dir_name: src_files["bindings"]}

    @abc.abstractmethod
    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        """
        Generate and build a python extension for the stencil computation.

        Returns the name and file path (as string) of the compiled extension ".so" module.
        """
        pass

    def make_extension(
        self, *, ir: Optional[gtir.Stencil] = None, uses_cuda: bool = False
    ) -> Tuple[str, str]:
        build_info = self.builder.options.build_info
        if build_info is not None:
            start_time = time.perf_counter()

        if not ir:
            ir = self.builder.gtir
        # Generate source
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_sources: Dict[str, Any] = self.make_extension_sources(ir=ir)
            gt_pyext_sources = {**gt_pyext_sources["computation"], **gt_pyext_sources["bindings"]}
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_sources = {
                key: gt_utils.NOTHING for key in self.PYEXT_GENERATOR_CLASS.TEMPLATE_FILES.keys()
            }

        if build_info is not None:
            next_time = time.perf_counter()
            build_info["codegen_time"] = next_time - start_time
            start_time = next_time

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **gt_backend.pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
                gt_version=2,
            ),
        )

        result = self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)

        if build_info is not None:
            build_info["build_time"] = time.perf_counter() - start_time

        return result

    def make_extension_sources(self, *, ir: gtir.Stencil) -> Dict[str, Dict[str, str]]:
        """Generate the source for the stencil independently from use case."""
        if "computation_src" in self.builder.backend_data:
            return self.builder.backend_data["computation_src"]
        class_name = self.pyext_class_name if self.builder.stencil_id else self.builder.options.name
        module_name = (
            self.pyext_module_name
            if self.builder.stencil_id
            else f"{self.builder.options.name}_pyext"
        )
        gt_pyext_generator = self.PYEXT_GENERATOR_CLASS(class_name, module_name, self)
        gt_pyext_sources = gt_pyext_generator(ir)
        final_ext = ".cu" if self.languages and self.languages["computation"] == "cuda" else ".cpp"
        comp_src = gt_pyext_sources["computation"]
        for key in [k for k in comp_src.keys() if k.endswith(".src")]:
            comp_src[key.replace(".src", final_ext)] = comp_src.pop(key)
        self.builder.backend_data["computation_src"] = gt_pyext_sources
        return gt_pyext_sources


class CUDAPyExtModuleGenerator(PyExtModuleGenerator):
    def generate_implementation(self) -> str:
        source = super().generate_implementation()
        if self.builder.options.backend_opts.get("device_sync", True):
            source += textwrap.dedent(
                """
                    cupy.cuda.Device(0).synchronize()
                """
            )
        return source

    def generate_imports(self) -> str:
        source = (
            textwrap.dedent(
                """
                import cupy
                """
            )
            + super().generate_imports()
        )
        return source


class GTCUDAPyModuleGenerator(CUDAPyExtModuleGenerator):
    def generate_pre_run(self) -> str:
        field_names = [
            key
            for key in self.args_data.field_info
            if self.args_data.field_info[key].access != AccessKind.NONE
        ]

        return "\n".join([f + ".host_to_device()" for f in field_names])

    def generate_post_run(self) -> str:
        output_field_names = [
            name
            for name, info in self.args_data.field_info.items()
            if info is not None and bool(info.access & AccessKind.WRITE)
        ]

        return "\n".join([f + "._set_device_modified()" for f in output_field_names])


def make_x86_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = iter(range(sum(mask)))
    if len(mask) < 3:
        layout: List[Optional[int]] = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask: List[Optional[int]] = [*mask[3:], *mask[:3]]
        layout = [next(ctr) if m else None for m in swapped_mask]

        layout = [*layout[-3:], *layout[:-3]]

    return tuple(layout)


def x86_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_x86_layout_map(field.mask)
    flattened_layout = [index for index in layout_map if index is not None]
    if len(field.strides) < len(flattened_layout):
        return False
    for dim in reversed(np.argsort(flattened_layout)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def gtcpu_is_compatible_type(field: "Storage") -> bool:
    return isinstance(field, np.ndarray)


def make_mc_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = reversed(range(sum(mask)))
    if len(mask) < 3:
        layout: List[Optional[int]] = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask: List[Optional[int]] = list(mask)
        tmp = swapped_mask[1]
        swapped_mask[1] = swapped_mask[2]
        swapped_mask[2] = tmp

        layout = [next(ctr) if m else None for m in swapped_mask]

        tmp = layout[1]
        layout[1] = layout[2]
        layout[2] = tmp

    return tuple(layout)


def mc_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_mc_layout_map(field.mask)
    flattened_layout = [index for index in layout_map if index is not None]
    if len(field.strides) < len(flattened_layout):
        return False
    for dim in reversed(np.argsort(flattened_layout)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def make_cuda_layout_map(mask: Tuple[int, ...]) -> Tuple[Optional[int], ...]:
    ctr = reversed(range(sum(mask)))
    return tuple([next(ctr) if m else None for m in mask])


def cuda_is_compatible_layout(field: "Storage") -> bool:
    stride = 0
    layout_map = make_cuda_layout_map(field.mask)
    flattened_layout = [index for index in layout_map if index is not None]
    if len(field.strides) < len(flattened_layout):
        return False
    for dim in reversed(np.argsort(flattened_layout)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def cuda_is_compatible_type(field: Any) -> bool:
    from gt4py.storage.storage import ExplicitlySyncedGPUStorage, GPUStorage

    return isinstance(field, (GPUStorage, ExplicitlySyncedGPUStorage))
