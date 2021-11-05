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

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

import dace

import gt4py.utils as gt_utils
from eve import NodeVisitor, codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import backend as gt_backend
from gt4py import gt_src_manager
from gt4py.backend import BaseGTBackend, CLIBackendMixin
from gt4py.backend.gt_backends import make_x86_layout_map, x86_is_compatible_layout
from gt4py.backend.gtc_backend.common import bindings_main_template, pybuffer_to_sid
from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gt4py.ir import StencilDefinition
from gtc import gtir, gtir_to_oir
from gtc.common import LevelMarker
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import array_dimensions
from gtc.passes.gtir_legacy_extents import compute_legacy_extents
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.caches import FillFlushToLocalKCaches
from gtc.passes.oir_optimizations.inlining import MaskInlining
from gtc.passes.oir_optimizations.mask_stmt_merging import MaskStmtMerging
from gtc.passes.oir_pipeline import DefaultPipeline


if TYPE_CHECKING:
    from gt4py.stencil_object import StencilObject


class GTCDaCeExtGenerator:
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, definition_ir: StencilDefinition) -> Dict[str, Dict[str, str]]:
        gtir = GtirPipeline(DefIRToGTIR.apply(definition_ir)).full()
        base_oir = gtir_to_oir.GTIRToOIR().visit(gtir)
        oir_pipeline = self.backend.builder.options.backend_opts.get(
            "oir_pipeline",
            DefaultPipeline(skip=[MaskStmtMerging, MaskInlining, FillFlushToLocalKCaches]),
        )
        oir = oir_pipeline.run(base_oir)
        sdfg = OirSDFGBuilder().visit(oir)
        sdfg.expand_library_nodes(recursive=True)
        sdfg.apply_strict_transformations(validate=True)

        implementation = DaCeComputationCodegen.apply(gtir, sdfg)
        bindings = DaCeBindingsCodegen.apply(
            gtir, sdfg, module_name=self.module_name, backend=self.backend
        )

        bindings_ext = ".cu" if self.backend.GT_BACKEND_T == "gpu" else ".cpp"
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {"bindings" + bindings_ext: bindings},
        }


class KOriginsVisitor(NodeVisitor):
    def visit_Stencil(self, node: gtir.Stencil):
        k_origins: Dict[str, int] = dict()
        self.generic_visit(node, k_origins=k_origins)
        return k_origins

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        self.generic_visit(node, interval=node.interval, **kwargs)

    def visit_FieldAccess(
        self, node: gtir.FieldAccess, *, k_origins, interval: gtir.Interval
    ) -> None:
        if interval.start.level == LevelMarker.START:
            candidate = max(0, -interval.start.offset - node.offset.k)
        else:
            candidate = 0
        k_origins[node.name] = max(k_origins.get(node.name, 0), candidate)


def compute_k_origins(node: gtir.Stencil) -> Dict[str, int]:
    return KOriginsVisitor().visit(node)


class DaCeComputationCodegen:

    template = as_mako(
        """
        auto ${name}(const std::array<gt::uint_t, 3>& domain) {
            return [domain](${",".join(functor_args)}) {
                const int __I = domain[0];
                const int __J = domain[1];
                const int __K = domain[2];
                ${name}_t dace_handle;
                auto allocator = gt::sid::make_cached_allocator(&std::make_unique<char[]>);
                ${"\\n".join(tmp_allocs)}
                __program_${name}(${",".join(["&dace_handle", *dace_args])});
            };
        }
        """
    )

    def generate_tmp_allocs(self, sdfg):
        fmt = "dace_handle.{name} = allocate(allocator, gt::meta::lazy::id<{dtype}>(), {size})();"
        return [
            fmt.format(
                name=f"__{sdfg.sdfg_id}_{name}", dtype=array.dtype.ctype, size=array.total_size
            )
            for name, array in sdfg.arrays.items()
            if array.transient and array.lifetime == dace.AllocationLifetime.Persistent
        ]

    @classmethod
    def apply(cls, gtir, sdfg: dace.SDFG):
        self = cls()
        code_objects = sdfg.generate_code()
        computations = code_objects[[co.title for co in code_objects].index("Frame")].clean_code
        lines = computations.split("\n")
        computations = "\n".join(lines[0:2] + lines[3:])  # remove import of not generated file
        computations = codegen.format_source("cpp", computations, style="LLVM")
        interface = cls.template.definition.render(
            name=sdfg.name,
            dace_args=self.generate_dace_args(gtir, sdfg),
            functor_args=self.generate_functor_args(sdfg),
            tmp_allocs=self.generate_tmp_allocs(sdfg),
        )
        generated_code = f"""#include <gridtools/sid/sid_shift_origin.hpp>
                             #include <gridtools/sid/allocator.hpp>
                             #include <gridtools/stencil/cartesian.hpp>
                             namespace gt = gridtools;
                             {computations}
                             {interface}
                             """
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code

    def __init__(self):
        self._unique_index = 0

    def generate_dace_args(self, gtir, sdfg):
        offset_dict: Dict[str, Tuple[int, int, int]] = {
            k: (-v[0][0], -v[1][0], -v[2][0]) for k, v in compute_legacy_extents(gtir).items()
        }
        k_origins = compute_k_origins(gtir)
        for name, origin in k_origins.items():
            offset_dict[name] = (offset_dict[name][0], offset_dict[name][1], origin)

        symbols = {f"__{var}": f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            if array.transient:
                decl = gtir.symtable_[name]
                layout_map = make_x86_layout_map(
                    tuple([1 if d else 0 for d in decl.dimensions] + [1 for dim in decl.data_dims])
                )
                expanded_shape = gt_utils.interpolate_mask(
                    array.shape, [m is not None for m in layout_map], default=None
                )
                index = 0
                strides = []
                # TODO (jdahm): This could use refactoring
                while True:
                    try:
                        shape_index = layout_map.index(index)
                        axis_size = expanded_shape[shape_index]
                        symbol_name = (
                            f"__{name}_{'IJK'[shape_index]}_stride"
                            if shape_index < 3
                            else f"__{name}_d{shape_index - 3}_stride"
                        )
                        symbols[symbol_name] = "*".join(strides) or "1"
                        strides.append(str(axis_size))
                        index += 1
                    except Exception:
                        break
            else:
                dims = [dim for dim, select in zip("IJK", array_dimensions(array)) if select]
                data_ndim = len(array.shape) - len(dims)

                # api field strides
                fmt = "gt::sid::get_stride<{dim}>(gt::sid::get_strides(__{name}_sid))"

                symbols.update(
                    {
                        f"__{name}_{dim}_stride": fmt.format(
                            dim=f"gt::stencil::dim::{dim.lower()}", name=name
                        )
                        for dim in dims
                    }
                )
                symbols.update(
                    {
                        f"__{name}_d{dim}_stride": fmt.format(
                            dim=f"gt::integral_constant<int, {3 + dim}>", name=name
                        )
                        for dim in range(data_ndim)
                    }
                )

                # api field pointers
                fmt = """gt::sid::multi_shifted(
                             gt::sid::get_origin(__{name}_sid)(),
                             gt::sid::get_strides(__{name}_sid),
                             std::array<gt::int_t, {ndim}>{{{origin}}}
                         )"""
                origin = tuple(
                    -offset_dict[name][idx]
                    for idx, var in enumerate("IJK")
                    if any(
                        dace.symbolic.pystr_to_symbolic(f"__{var}") in s.free_symbols
                        for s in array.shape
                        if hasattr(s, "free_symbols")
                    )
                )
                symbols[name] = fmt.format(
                    name=name, ndim=len(array.shape), origin=",".join(str(o) for o in origin)
                )
        # the remaining arguments are variables and can be passed by name
        for sym in sdfg.signature_arglist(with_types=False, for_call=True):
            if sym not in symbols:
                symbols[sym] = sym

        # return strings in order of sdfg signature
        return [symbols[s] for s in sdfg.signature_arglist(with_types=False, for_call=True)]

    def generate_functor_args(self, sdfg: dace.SDFG):
        res = []
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            res.append(f"auto && __{name}_sid")
        for name, dtype in ((n, d) for n, d in sdfg.symbols.items() if not n.startswith("__")):
            res.append(dtype.as_arg(name))
        return res


class DaCeBindingsCodegen:
    def __init__(self, backend):
        self.backend = backend
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    mako_template = bindings_main_template()

    def generate_entry_params(self, gtir: gtir.Stencil, sdfg: dace.SDFG):
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
            elif name in sdfg.symbols and not name.startswith("__"):
                assert name in sdfg.symbols
                res[name] = "{dtype} {name}".format(dtype=sdfg.symbols[name].ctype, name=name)
        return list(res[node.name] for node in gtir.params if node.name in res)

    def generate_sid_params(self, sdfg: dace.SDFG):
        res = []
        import dace.data

        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            domain_dim_flags = tuple(
                True
                if any(
                    dace.symbolic.pystr_to_symbolic(f"__{dim.upper()}") in s.free_symbols
                    for s in array.shape
                    if hasattr(s, "free_symbols")
                )
                else False
                for dim in "ijk"
            )
            data_ndim = len(array.shape) - sum(array_dimensions(array))
            sid_def = pybuffer_to_sid(
                name=name,
                ctype=array.dtype.ctype,
                domain_dim_flags=domain_dim_flags,
                data_ndim=data_ndim,
                stride_kind_index=self.unique_index(),
                backend=self.backend,
            )

            res.append(sid_def)
        # pass scalar parameters as variables
        for name in (n for n in sdfg.symbols.keys() if not n.startswith("__")):
            res.append(name)
        return res

    def generate_sdfg_bindings(self, gtir, sdfg, module_name):

        return self.mako_template.render_values(
            name=sdfg.name,
            module_name=module_name,
            entry_params=self.generate_entry_params(gtir, sdfg),
            sid_params=self.generate_sid_params(sdfg),
        )

    @classmethod
    def apply(cls, gtir: gtir.Stencil, sdfg: dace.SDFG, module_name: str, *, backend) -> str:
        generated_code = cls(backend).generate_sdfg_bindings(gtir, sdfg, module_name=module_name)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


@gt_backend.register
class GTCDaceBackend(BaseGTBackend, CLIBackendMixin):
    """DaCe python backend using gtc."""

    name = "gtc:dace"
    GT_BACKEND_T = "dace"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
        "is_compatible_type": x86_is_compatible_layout,
    }

    options = BaseGTBackend.GT_BACKEND_OPTS
    PYEXT_GENERATOR_CLASS = GTCDaCeExtGenerator  # type: ignore
    USE_LEGACY_TOOLCHAIN = False

    def generate_extension(self) -> Tuple[str, str]:
        return self.make_extension(gt_version=2, ir=self.builder.definition_ir, uses_cuda=False)

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
