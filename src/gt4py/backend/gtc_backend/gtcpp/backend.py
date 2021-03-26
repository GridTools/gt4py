# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type
import dace
import gtc.dace_to_oir as dace_to_oir
import gtc.utils as gtc_utils
from eve import codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin
from gt4py.backend.gt_backends import (
    GTCUDAPyModuleGenerator,
    cuda_is_compatible_layout,
    cuda_is_compatible_type,
    cuda_layout,
    gtcpu_is_compatible_type,
    make_mc_layout_map,
    make_x86_layout_map,
    mc_is_compatible_layout,
    x86_is_compatible_layout,
)
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir_to_oir
from gtc.common import DataType
from gtc.gtcpp import gtcpp, gtcpp_codegen, oir_to_gtcpp
from gtc.oir_to_dace import OirSDFGBuilder
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast
from gtc.passes.oir_optimizations.caches import (
    IJCacheDetection,
    KCacheDetection,
    PruneKCacheFills,
    PruneKCacheFlushes,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import GreedyMerging, OnTheFlyMerging
from gtc.passes.oir_optimizations.pruning import NoFieldAccessPruning
from gtc.passes.oir_optimizations.temporaries import (
    LocalTemporariesToScalars,
    WriteBeforeReadTemporariesToScalars,
)
from gtc.passes.oir_optimizations.vertical_loop_merging import AdjacentLoopMerging
from gtc import oir

if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject

from gt4py.ir import StencilDefinition


class GTCGTExtGenerator:
    def __init__(self, class_name, module_name, gt_backend_t, options):
        self.class_name = class_name
        self.module_name = module_name
        self.gt_backend_t = gt_backend_t
        self.options = options

    def __call__(self, definition_ir: StencilDefinition) -> Dict[str, Dict[str, str]]:
        gtir = DefIRToGTIR.apply(definition_ir)
        gtir_without_unused_params = prune_unused_parameters(gtir)
        dtype_deduced = resolve_dtype(gtir_without_unused_params)
        upcasted = upcast(dtype_deduced)
        oir = gtir_to_oir.GTIRToOIR().visit(upcasted)
        oir = self._optimize_oir(oir)
        oir = gtir_to_oir.oir_iteration_space_computation(oir)
        sdfg = OirSDFGBuilder.build(oir.name, oir)
        sdfg.save(definition_ir.name + "_toplevel.sdfg")

        import copy

        sdfg2 = copy.deepcopy(sdfg)
        sdfg2.expand_library_nodes(recursive=False)
        sdfg2.save(definition_ir.name + "_expanded.sdfg")
        sdfg2.expand_library_nodes(recursive=False)
        sdfg2.save(definition_ir.name + "_expanded2.sdfg")
        for n, _ in sdfg2.all_nodes_recursive():
            if isinstance(n, dace.nodes.MapEntry):
                n.map.schedule=dace.ScheduleType.Sequential
        sdfg2.validate()
        # sdfg2.apply_strict_transformations(validate=False)

        # from dace.transformation.interstate import StateFusion, InlineSDFG
        # for node, _ in sdfg2.all_nodes_recursive():
        #     import dace
        #     if isinstance(node, dace.nodes.NestedSDFG):
        #         node.sdfg.apply_transformations_repeated([InlineSDFG], strict=True, validate=False)
        #     sdfg2.apply_transformations_repeated([InlineSDFG], strict=True, validate=False)
        #     # sdfg2.apply_transformations_repeated(StateFusion, strict=True, validate=False)

        sdfg2.save(definition_ir.name + "_strict.sdfg")
        sdfg2.validate()
        # sdfg2.compile()
        oir = dace_to_oir.convert(sdfg)

        # gtcpp = oir_to_gtcpp.OIRToGTCpp().visit(oir)
        # implementation = gtcpp_codegen.GTCppCodegen.apply(gtcpp, gt_backend_t=self.gt_backend_t)
        code_objects = sdfg2.generate_code()
        implementation = code_objects[[co.title for co in code_objects].index("Frame")].clean_code
        lines = implementation.split("\n")
        implementation = "\n".join(lines[0:2] + lines[3:])
        implementation = codegen.format_source("cpp", implementation, style="LLVM")
        from gtc.gtir_to_oir import oir_field_boundary_computation
        origins = {k:(-v.i_interval.start.offset, -v.j_interval.start.offset, 0) for k,v in oir_field_boundary_computation(oir).items()}
        bindings = DaCeBindingsCodegen.apply(oir,
            sdfg2, module_name=self.module_name, origins=origins
        )

        bindings_ext = ".cu" if self.gt_backend_t == "gpu" else ".cpp"
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings" + bindings_ext: bindings},
        }

    def _optimize_oir(self, oir):
        oir = GreedyMerging().visit(oir)
        oir = AdjacentLoopMerging().visit(oir)
        oir = LocalTemporariesToScalars().visit(oir)
        oir = WriteBeforeReadTemporariesToScalars().visit(oir)
        oir = OnTheFlyMerging().visit(oir)
        oir = NoFieldAccessPruning().visit(oir)
        oir = IJCacheDetection().visit(oir)
        oir = KCacheDetection().visit(oir)
        oir = PruneKCacheFills().visit(oir)
        oir = PruneKCacheFlushes().visit(oir)
        return oir


class DaCeBindingsCodegen:
    def __init__(self):
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index


    mako_template = as_mako(
"""
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gridtools/storage/adapter/python_sid_adapter.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#include <gridtools/sid/sid_shift_origin.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include "computation.hpp"
namespace gt = gridtools;
namespace py = ::pybind11;
%if len(entry_params) > 0:

class ${name}_functor {
  const int __I;
  const int __J;
  const int __K;
  ${name}_t *dace_handle;

public:
  ${name}_functor(std::array<gt::uint_t, 3> domain)
      : __I(domain[0]), __J(domain[1]), __K(domain[2]),
        dace_handle(__dace_init_${name}(${zeros_init_func})){};
  ~${name}_functor() {
    __dace_exit_${name}(dace_handle);
  };
  void operator()(${functor_args}) {
  
    ${get_strides_and_ptr}
    
    __program_${name}(
        dace_handle, ${dace_args});
        
  }
};

std::tuple<gt::uint_t, gt::uint_t, gt::uint_t> last_size;
${name}_functor *${name}_ptr(nullptr);
${name}_functor & get_${name}(std::array<gt::uint_t, 3> domain) {
  auto this_size = std::tuple_cat(domain);
  if (${name}_ptr!=nullptr || this_size!=last_size){
      delete ${name}_ptr;
      ${name}_ptr = new ${name}_functor(domain);
  }
  return *${name}_ptr;
}

PYBIND11_MODULE(${module_name}, m) {
    m.def("run_computation", [](std::array<gt::uint_t, 3> domain,
    ${entry_params},
    py::object exec_info){
        if (!exec_info.is(py::none()))
        {
            auto exec_info_dict = exec_info.cast<py::dict>();
            exec_info_dict["run_cpp_start_time"] = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
        }

        get_${name}(domain)(${sid_params});

        if (!exec_info.is(py::none()))
        {
            auto exec_info_dict = exec_info.cast<py::dict>();
            exec_info_dict["run_cpp_end_time"] = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
        }

    }, "Runs the given computation");}
%endif
"""
    )

    def generate_strides_and_ptr(self, sdfg, offset_dict):
        res = []
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            if name in offset_dict:
                offset = ", ".join(str(-o) for o in offset_dict[name])
            else:
                offset = "0, 0, 0"
            # res.append(f"""auto __{name}_outer =gt::sid::shift_sid_origin(__{name}_sid, std::array<gt::int_t, 3>({{{offset}}}));
            #            auto {name} = gt::sid::sid_get_origin(__{name}_outer)();
            #            auto __{name}_strides = gt::sid::sid_get_strides(__{name}_sid);
            #            int __{name}_I_stride = __{name}_strides[0];
            #            int __{name}_J_stride = __{name}_strides[1];
            #            int __{name}_K_stride = __{name}_strides[2];"""
            # )
            dtype=array.dtype.ctype
            res.append(f"""
                       auto __{name}_outer =gt::sid::shift_sid_origin(__{name}_sid, std::array<gt::int_t, 3>{{0, 0, 0}});
                       {dtype}* {name} = gt::sid::sid_get_origin(__{name}_outer)();
                       auto __{name}_strides = gt::sid::sid_get_strides(__{name}_sid);
                       int __{name}_I_stride = __{name}_strides[0];
                       int __{name}_J_stride = __{name}_strides[1];
                       int __{name}_K_stride = __{name}_strides[2];
                       {name} -= ({"+".join(f"__{name}_{var}_stride*({offset_dict[name][idx]})" for idx, var in enumerate("IJK"))});"""
            )
        return "\n".join(res)

    def generate_entry_params(self, oir: oir.Stencil, sdfg: dace.SDFG):
        res = {}
        import dace.data
        for name in sdfg.signature_arglist(with_types=False, for_call=True):
            if name in sdfg.arrays:
                data = sdfg.arrays[name]
                assert isinstance(data, dace.data.Array)
                res[name] = "py::buffer {name}, std::array<gt::uint_t,{ndim}> {name}_origin".format(
                    name=name,
                    ndim=len(data.shape),
                )
            elif name in sdfg.symbols and not name.startswith('__'):
                assert name in sdfg.symbols
                res[name] = "{dtype} {name}".format(dtype=sdfg.symbols[name].ctype, name=name )
        return ",".join(res[node.name] for node in oir.params)

    def generate_sid_params(self, sdfg: dace.SDFG):
        res = []
        import dace.data
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            num_dims = len(array.shape)
            sid_def = """gt::as_{sid_type}<{dtype}, {num_dims},
                std::integral_constant<int, {unique_index}>>({name})""".format(
                sid_type="cuda_sid" if array.storage in [dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared] else "sid",
                name=name,
                dtype=array.dtype.ctype,
                unique_index=self.unique_index(),
                num_dims=num_dims,
            )
            res.append("gt::sid::shift_sid_origin({sid_def}, {name}_origin)".format(
                sid_def=sid_def,
                name=name,
            ))
        for name in (n for n in sdfg.symbols.keys() if not n.startswith('__')):
            res.append(name)
        return ", ".join(res)

    def generate_transient_strides(self, sdfg):
        symbols = {f"__{var}":f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            if array.transient:
                symbols[f"__{name}_K_stride"] = "1"
                symbols[f"__{name}_J_stride"] = str(array.shape[2])
                symbols[f"__{name}_I_stride"] = str(array.shape[1]*array.shape[2])
        for sym in sdfg.symbols.keys():
            if sym not in symbols:
                symbols[sym] = "0"
        return ",".join(symbols[s] for s in sdfg.signature_arglist(with_types=False, for_call=True) if s in symbols)
    def generate_dace_args(self, sdfg):
        symbols = {f"__{var}":f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            if array.transient:
                symbols[f"__{name}_K_stride"] = "1"
                symbols[f"__{name}_J_stride"] = str(array.shape[2])
                symbols[f"__{name}_I_stride"] = str(array.shape[1]*array.shape[2])
        for sym in sdfg.signature_arglist(with_types=False, for_call=True):
            if sym not in symbols:
                symbols[sym] = sym
        return ",".join(symbols[s] for s in sdfg.signature_arglist(with_types=False, for_call=True))

    def generate_functor_args(self, sdfg: dace.SDFG):
        res = []
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            res.append(f"auto &&__{name}_sid")
        for name, dtype in ((n, d) for n, d in sdfg.symbols.items() if not n.startswith('__')):
            res.append(dtype.as_arg(name))
        return ", ".join(res)

    def generate_sdfg_bindings(self, oir, sdfg, module_name, origins):

        return self.mako_template.render_values(
            name=sdfg.name,
            module_name=module_name,
            get_strides_and_ptr=self.generate_strides_and_ptr(sdfg, origins),
            dace_args=self.generate_dace_args(sdfg),
            entry_params=self.generate_entry_params(oir, sdfg),
            sid_params=self.generate_sid_params(sdfg),
            functor_args=self.generate_functor_args(sdfg),
            zeros_init_func=self.generate_transient_strides(sdfg),
        )
    @classmethod
    def apply(cls, oir: oir.Stencil, sdfg: dace.SDFG, module_name: str, origins) -> str:
        generated_code = cls().generate_sdfg_bindings(oir, sdfg, module_name=module_name, origins=origins)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


class GTCGTBaseBackend(BaseGTBackend, CLIBackendMixin):
    options = BaseGTBackend.GT_BACKEND_OPTS
    PYEXT_GENERATOR_CLASS = GTCGTExtGenerator  # type: ignore

    def _generate_extension(self, uses_cuda: bool) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=uses_cuda)

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources(2) and not gt_src_manager.install_gt_sources(2):
            raise RuntimeError("Missing GridTools sources.")

        pyext_module_name: Optional[str]
        pyext_file_path: Optional[str]

        # TODO(havogt) add bypass if computation has no effect
        pyext_module_name, pyext_file_path = self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module(
            pyext_module_name=pyext_module_name,
            pyext_file_path=pyext_file_path,
        )


@gt_backend.register
class GTCGTCpuIfirstBackend(GTCGTBaseBackend):
    """GridTools python backend using gtc."""

    name = "gtc:gt:cpu_ifirst"
    GT_BACKEND_T = "cpu_ifirst"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 8,
        "device": "cpu",
        "layout_map": make_mc_layout_map,
        "is_compatible_layout": mc_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=False)


@gt_backend.register
class GTCGTCpuKfirstBackend(GTCGTBaseBackend):
    """GridTools python backend using gtc."""

    name = "gtc:gt:cpu_kfirst"
    GT_BACKEND_T = "cpu_kfirst"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": gtcpu_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=False)


@gt_backend.register
class GTCGTGpuBackend(GTCGTBaseBackend):
    """GridTools python backend using gtc."""

    MODULE_GENERATOR_CLASS = GTCUDAPyModuleGenerator
    name = "gtc:gt:gpu"
    GT_BACKEND_T = "gpu"
    languages = {"computation": "cuda", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": cuda_layout,
        "is_compatible_layout": cuda_is_compatible_layout,
        "is_compatible_type": cuda_is_compatible_type,
    }

    def generate_extension(self, **kwargs: Any) -> Tuple[str, str]:
        return super()._generate_extension(uses_cuda=True)
