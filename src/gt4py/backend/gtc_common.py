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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import gtc.utils
import gtc.utils as gtc_utils
from eve.codegen import MakoTemplate as as_mako
from gt4py import backend as gt_backend
from gt4py import utils as gt_utils
from gt4py.backend import Backend
from gt4py.backend.module_generator import BaseModuleGenerator, ModuleData
from gtc import gtir
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_pipeline import OirPipeline


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder
    from gt4py.stencil_object import StencilObject


def _get_unit_stride_dim(backend, domain_dim_flags, data_ndim):
    make_layout_map = backend.storage_info["layout_map"]
    dimensions = list(gtc.utils.dimension_flags_to_names(domain_dim_flags).upper()) + [
        str(d) for d in range(data_ndim)
    ]
    layout_map = [x for x in make_layout_map(dimensions) if x is not None]
    return layout_map.index(max(layout_map))


def pybuffer_to_sid(
    *,
    name: str,
    ctype: str,
    domain_dim_flags: Tuple[bool, bool, bool],
    data_ndim: int,
    stride_kind_index: int,
    backend: Backend,
):
    domain_ndim = domain_dim_flags.count(True)
    sid_ndim = domain_ndim + data_ndim

    as_sid = "as_cuda_sid" if backend.storage_info["device"] == "gpu" else "as_sid"

    sid_def = """gt::{as_sid}<{ctype}, {sid_ndim},
        gt::integral_constant<int, {unique_index}>>({name})""".format(
        name=name,
        ctype=ctype,
        unique_index=stride_kind_index,
        sid_ndim=sid_ndim,
        as_sid=as_sid,
    )
    sid_def = "gt::sid::shift_sid_origin({sid_def}, {name}_origin)".format(
        sid_def=sid_def,
        name=name,
    )
    if domain_ndim != 3:
        gt_dims = [
            f"gt::stencil::dim::{dim}"
            for dim in gtc_utils.dimension_flags_to_names(domain_dim_flags)
        ]
        if data_ndim:
            gt_dims += [f"gt::integral_constant<int, {3 + dim}>" for dim in range(data_ndim)]
        sid_def = "gt::sid::rename_numbered_dimensions<{gt_dims}>({sid_def})".format(
            gt_dims=", ".join(gt_dims), sid_def=sid_def
        )

    return sid_def


def bindings_main_template():
    return as_mako(
        """
        #include <chrono>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <gridtools/stencil/cartesian.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/sid/sid_shift_origin.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include "computation.hpp"
        namespace gt = gridtools;
        namespace py = ::pybind11;
        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", [](
            ${','.join(["std::array<gt::uint_t, 3> domain", *entry_params, 'py::object exec_info'])}
            ){
                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_start_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
                }

                ${name}(domain)(${','.join(sid_params)});

                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_end_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
                }

            }, "Runs the given computation");}
        """
    )


def gtir_is_not_empty(pipeline: GtirPipeline) -> bool:
    node = pipeline.full()
    return bool(node.walk_values().if_isinstance(gtir.ParAssignStmt).to_list())


def gtir_has_effect(pipeline: GtirPipeline) -> bool:
    return True


class PyExtModuleGenerator(BaseModuleGenerator):
    """Module Generator for use with backends that generate c++ python extensions."""

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
        return gtir_is_not_empty(self.builder.gtir_pipeline)

    def generate_imports(self) -> str:
        source = [*super().generate_imports().splitlines(), "from gt4py import utils as gt_utils"]
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
        return gtir_has_effect(self.builder.gtir_pipeline)

    def generate_implementation(self) -> str:
        ir = self.builder.gtir
        sources = gt_utils.text.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args: List[str] = []
        for decl in ir.params:
            args.append(decl.name)
            if isinstance(decl, gtir.FieldDecl):
                args.append("list(_origin_['{}'])".format(decl.name))

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


class BackendCodegen:
    TEMPLATE_FILES: Dict[str, str]

    @abc.abstractmethod
    def __init__(self, class_name: str, module_name: str, backend: Any):
        pass

    @abc.abstractmethod
    def __call__(self, ir: gtir.Stencil) -> Dict[str, Dict[str, str]]:
        """Return a dict with the keys 'computation' and 'bindings' to dicts of filenames to source."""
        pass


class BaseGTBackend(gt_backend.BasePyExtBackend, gt_backend.CLIBackendMixin):

    GT_BACKEND_OPTS: Dict[str, Dict[str, Any]] = {
        "add_profile_info": {"versioning": True, "type": bool},
        "clean": {"versioning": False, "type": bool},
        "debug_mode": {"versioning": True, "type": bool},
        "verbose": {"versioning": False, "type": bool},
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
    }

    GT_BACKEND_T: str

    MODULE_GENERATOR_CLASS = PyExtModuleGenerator

    PYEXT_GENERATOR_CLASS: Type[BackendCodegen]

    @abc.abstractmethod
    def generate(self) -> Type["StencilObject"]:
        pass

    def generate_computation(self) -> Dict[str, Union[str, Dict]]:
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(stencil_ir=self.builder.gtir)
        return {dir_name: src_files["computation"]}

    def generate_bindings(
        self, language_name: str, *, stencil_ir: gtir.Stencil = None
    ) -> Dict[str, Union[str, Dict]]:
        if not stencil_ir:
            stencil_ir = self.builder.gtir
        if language_name != "python":
            return super().generate_bindings(language_name)
        dir_name = f"{self.builder.options.name}_src"
        src_files = self.make_extension_sources(stencil_ir=stencil_ir)
        return {dir_name: src_files["bindings"]}

    @abc.abstractmethod
    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        """
        Generate and build a python extension for the stencil computation.

        Returns the name and file path (as string) of the compiled extension ".so" module.
        """
        pass

    def make_extension(
        self, *, stencil_ir: Optional[gtir.Stencil] = None, uses_cuda: bool = False
    ) -> Tuple[str, str]:
        build_info = self.builder.options.build_info
        if build_info is not None:
            start_time = time.perf_counter()

        if not stencil_ir:
            stencil_ir = self.builder.gtir
        # Generate source
        gt_pyext_files: Dict[str, Any]
        gt_pyext_sources: Dict[str, Any]
        if not self.builder.options._impl_opts.get("disable-code-generation", False):
            gt_pyext_files = self.make_extension_sources(stencil_ir=stencil_ir)
            gt_pyext_sources = {**gt_pyext_files["computation"], **gt_pyext_files["bindings"]}
        else:
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_files = {}
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

    def make_extension_sources(self, *, stencil_ir: gtir.Stencil) -> Dict[str, Dict[str, str]]:
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
        gt_pyext_sources = gt_pyext_generator(stencil_ir)
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


def _permute_layout_to_dimensions(
    layout: Sequence[int], dimensions: Tuple[str, ...]
) -> Tuple[int, ...]:
    data_dims = [int(d) for d in dimensions if d.isdigit()]
    canonical_dimensions = [d for d in "IJK" if d in dimensions] + [
        str(d) for d in sorted(data_dims)
    ]
    res_layout = []
    for d in dimensions:
        res_layout.append(layout[canonical_dimensions.index(d)])
    return tuple(res_layout)


def make_gtcpu_kfirst_layout_map(dimensions: Tuple[str, ...]) -> Tuple[int, ...]:
    layout = [i for i in range(len(dimensions))]
    naxes = sum(dim in dimensions for dim in "IJK")
    layout = [*layout[-naxes:], *layout[:-naxes]]
    return _permute_layout_to_dimensions([lt for lt in layout if lt is not None], dimensions)


def make_gtcpu_ifirst_layout_map(dimensions: Tuple[str, ...]) -> Tuple[int, ...]:
    ctr = reversed(range(len(dimensions)))
    layout = [next(ctr) for dim in "IJK" if dim in dimensions] + list(ctr)
    if "K" in dimensions and "J" in dimensions:
        if "I" in dimensions:
            layout = [layout[0], layout[2], layout[1], *layout[3:]]
        else:
            layout = [layout[1], layout[0], *layout[2:]]
    return _permute_layout_to_dimensions(layout, dimensions)


def make_cuda_layout_map(dimensions: Tuple[str, ...]) -> Tuple[Optional[int], ...]:
    layout = tuple(reversed(range(len(dimensions))))
    return _permute_layout_to_dimensions(layout, dimensions)
