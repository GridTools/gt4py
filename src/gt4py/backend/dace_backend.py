# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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

import copy
import os
import pathlib
import re
import textwrap
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

import dace
import numpy as np
from dace.sdfg.utils import inline_sdfgs
from dace.serialize import dumps

from eve import codegen
from eve.codegen import MakoTemplate as as_mako
from gt4py import gt_src_manager
from gt4py.backend.base import CLIBackendMixin, register
from gt4py.backend.gtc_common import (
    BackendCodegen,
    BaseGTBackend,
    GTCUDAPyModuleGenerator,
    PyExtModuleGenerator,
    bindings_main_template,
    cuda_is_compatible_type,
    pybuffer_to_sid,
)
from gt4py.backend.module_generator import make_args_data_from_gtir
from gt4py.utils import shash
from gtc import common, gtir
from gtc.dace.nodes import StencilComputation
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import array_dimensions, layout_maker_factory, replace_strides
from gtc.gtir_to_oir import GTIRToOIR
from gtc.passes.gtir_k_boundary import compute_k_boundary
from gtc.passes.gtir_pipeline import GtirPipeline
from gtc.passes.oir_optimizations.inlining import MaskInlining
from gtc.passes.oir_optimizations.utils import compute_fields_extents
from gtc.passes.oir_pipeline import DefaultPipeline


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder
    from gt4py.stencil_object import StencilObject


def _serialize_sdfg(sdfg: dace.SDFG):
    return dumps(sdfg)


def _specialize_transient_strides(sdfg: dace.SDFG, layout_map):
    repldict = replace_strides(
        [array for array in sdfg.arrays.values() if array.transient],
        layout_map,
    )
    sdfg.replace_dict(repldict)
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                for k, v in repldict.items():
                    if k in node.symbol_mapping:
                        node.symbol_mapping[k] = v
    for k in repldict.keys():
        if k in sdfg.symbols:
            sdfg.remove_symbol(k)


def _to_device(sdfg: dace.SDFG, device: str) -> None:
    """Update sdfg in place."""
    if device == "gpu":
        for array in sdfg.arrays.values():
            array.storage = dace.StorageType.GPU_Global
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, StencilComputation):
                node.device = dace.DeviceType.GPU


def _pre_expand_trafos(gtir_pipeline: GtirPipeline, sdfg: dace.SDFG, layout_map):

    args_data = make_args_data_from_gtir(gtir_pipeline)

    # stencils without effect
    if all(info is None for info in args_data.field_info.values()):
        sdfg = dace.SDFG(gtir_pipeline.gtir.name)
        sdfg.add_state(gtir_pipeline.gtir.name)
        return sdfg

    for array in sdfg.arrays.values():
        if array.transient:
            array.lifetime = dace.AllocationLifetime.Persistent

    sdfg.simplify(validate=False)

    for node, _ in filter(
        lambda n: isinstance(n[0], StencilComputation), sdfg.all_nodes_recursive()
    ):
        expansion_priority = []
        if node.has_splittable_regions():
            expansion_priority.append(
                [
                    "Sections",
                    "Stages",
                    "J",
                    "I",
                    "K",
                ]
            )
        if node.oir_node.loop_order == common.LoopOrder.PARALLEL:
            expansion_priority.extend(
                [
                    ["Sections", "Stages", "K", "J", "I"],
                    ["TileJ", "TileI", "Sections", "KMap", "Stages", "JMap", "IMap"],
                ]
            )
        else:
            expansion_priority.extend(
                [
                    ["J", "I", "Sections", "Stages", "K"],
                    ["TileJ", "TileI", "Sections", "KLoop", "Stages", "JMap", "IMap"],
                ]
            )
        is_set = False
        for exp in expansion_priority:
            try:
                node.expansion_specification = exp
                is_set = True
            except ValueError:
                continue
            else:
                break
        if not is_set:
            raise ValueError("No expansion compatible")
    _specialize_transient_strides(sdfg, layout_map=layout_map)
    return sdfg


def _post_expand_trafos(sdfg: dace.SDFG):
    # DaCe "standard" clean-up transformations
    sdfg.simplify(validate=False)

    # Only has effect if schedule is CPU_Multicore,
    # setting node.collapse causes the omp parallel statement to include collapse(n)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.nodes.MapEntry):
            node.collapse = len(node.range)


def _sdfg_add_arrays_and_edges(
    field_info, wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs, origins
):

    for name, array in inner_sdfg.arrays.items():
        if isinstance(array, dace.data.Array) and not array.transient:
            axes = field_info[name].axes

            shape = [f"__{name}_{axis}_size" for axis in axes] + [
                d for d in field_info[name].data_dims
            ]

            wrapper_sdfg.add_array(
                name, dtype=array.dtype, strides=array.strides, shape=shape, storage=array.storage
            )
            if isinstance(origins, tuple):
                origin = [o for a, o in zip("IJK", origins) if a in axes]
            else:
                origin = origins.get(name, origins.get("_all_", None))
                if len(origin) == 3:
                    origin = [o for a, o in zip("IJK", origin) if a in axes]

            ranges = [
                (o - max(0, e), o - max(0, e) + s - 1, 1)
                for o, e, s in zip(
                    origin,
                    field_info[name].boundary.lower_indices,
                    inner_sdfg.arrays[name].shape,
                )
            ]
            ranges += [(0, d, 1) for d in field_info[name].data_dims]
            if name in inputs:
                state.add_edge(
                    state.add_read(name),
                    None,
                    nsdfg,
                    name,
                    dace.Memlet(name, subset=dace.subsets.Range(ranges)),
                )
            if name in outputs:
                state.add_edge(
                    nsdfg,
                    name,
                    state.add_write(name),
                    None,
                    dace.Memlet(name, subset=dace.subsets.Range(ranges)),
                )


def _sdfg_specialize_symbols(wrapper_sdfg, domain: Tuple[int, ...]):
    ival, jval, kval = domain[0], domain[1], domain[2]
    for sdfg in wrapper_sdfg.all_sdfgs_recursive():
        if sdfg.parent_nsdfg_node is not None:
            symmap = sdfg.parent_nsdfg_node.symbol_mapping

            if "__I" in symmap:
                ival = symmap["__I"]
                del symmap["__I"]
            if "__J" in symmap:
                jval = symmap["__J"]
                del symmap["__J"]
            if "__K" in symmap:
                kval = symmap["__K"]
                del symmap["__K"]

        sdfg.replace_dict({"__I": ival, "__J": jval, "__K": kval})
        if "__I" in sdfg.symbols:
            sdfg.remove_symbol("__I")
        if "__J" in sdfg.symbols:
            sdfg.remove_symbol("__J")
        if "__K" in sdfg.symbols:
            sdfg.remove_symbol("__K")

        for val in ival, jval, kval:
            sym = dace.symbolic.pystr_to_symbolic(val)
            for fsym in sym.free_symbols:
                if sdfg.parent_nsdfg_node is not None:
                    sdfg.parent_nsdfg_node.symbol_mapping[str(fsym)] = fsym
                if str(fsym) not in sdfg.symbols:
                    if str(fsym) in sdfg.parent_sdfg.symbols:
                        sdfg.add_symbol(str(fsym), stype=sdfg.parent_sdfg.symbols[str(fsym)])
                    else:
                        sdfg.add_symbol(str(fsym), stype=dace.dtypes.int32)


def freeze_origin_domain_sdfg(inner_sdfg, arg_names, field_info, *, origin, domain):
    wrapper_sdfg = dace.SDFG("frozen_" + inner_sdfg.name)
    state = wrapper_sdfg.add_state("frozen_" + inner_sdfg.name + "_state")

    inputs = set()
    outputs = set()
    for inner_state in inner_sdfg.nodes():
        for node in inner_state.nodes():
            if (
                not isinstance(node, dace.nodes.AccessNode)
                or inner_sdfg.arrays[node.data].transient
            ):
                continue
            if node.has_reads(inner_state):
                inputs.add(node.data)
            if node.has_writes(inner_state):
                outputs.add(node.data)

    nsdfg = state.add_nested_sdfg(inner_sdfg, None, inputs, outputs)

    _sdfg_add_arrays_and_edges(
        field_info, wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs, origins=origin
    )

    # in special case of empty domain, remove entire SDFG.
    if any(d == 0 for d in domain):
        states = wrapper_sdfg.states()
        assert len(states) == 1
        for node in states[0].nodes():
            state.remove_node(node)

    # make sure that symbols are passed throught o inner sdfg
    for symbol in nsdfg.sdfg.free_symbols:
        if symbol not in wrapper_sdfg.symbols:
            wrapper_sdfg.add_symbol(symbol, nsdfg.sdfg.symbols[symbol])

    # Try to inline wrapped SDFG before symbols are specialized to avoid extra views
    inline_sdfgs(wrapper_sdfg)

    _sdfg_specialize_symbols(wrapper_sdfg, domain)

    for _, _, array in wrapper_sdfg.arrays_recursive():
        if array.transient:
            array.lifetime = dace.dtypes.AllocationLifetime.SDFG

    wrapper_sdfg.arg_names = arg_names

    return wrapper_sdfg


class SDFGManager:

    _loaded_sdfgs: Dict[str, dace.SDFG] = dict()

    def __init__(self, builder):
        self.builder = builder

    @staticmethod
    def _strip_history(sdfg):
        # strip history from SDFG for faster save/load
        for tmp_sdfg in sdfg.all_sdfgs_recursive():
            tmp_sdfg.transformation_hist = []
            tmp_sdfg.orig_sdfg = None

    @staticmethod
    def _save_sdfg(sdfg, path):
        SDFGManager._strip_history(sdfg)
        sdfg.save(path)

    def _unexpanded_sdfg(self):
        filename = self.builder.module_name + ".sdfg"
        path = (
            pathlib.Path(os.path.relpath(self.builder.module_path.parent, pathlib.Path.cwd()))
            / filename
        )

        if path not in SDFGManager._loaded_sdfgs:

            try:
                sdfg = dace.SDFG.from_file(path)
            except FileNotFoundError:

                base_oir = GTIRToOIR().visit(self.builder.gtir)
                oir_pipeline = self.builder.options.backend_opts.get(
                    "oir_pipeline", DefaultPipeline(skip=[MaskInlining])
                )
                oir_node = oir_pipeline.run(base_oir)
                sdfg = OirSDFGBuilder().visit(oir_node)

                _to_device(sdfg, self.builder.backend.storage_info["device"])
                _pre_expand_trafos(
                    self.builder.gtir_pipeline,
                    sdfg,
                    self.builder.backend.storage_info["layout_map"],
                )
                self._save_sdfg(sdfg, path)
            self._loaded_sdfgs[path] = sdfg

        return SDFGManager._loaded_sdfgs[path]

    def unexpanded_sdfg(self):
        return copy.deepcopy(self._unexpanded_sdfg())

    def _expanded_sdfg(self):
        sdfg = self._unexpanded_sdfg()
        sdfg.expand_library_nodes()
        _post_expand_trafos(sdfg)
        return sdfg

    def expanded_sdfg(self):
        return copy.deepcopy(self._expanded_sdfg())

    def _frozen_sdfg(self, *, origin: Dict[str, Tuple[int, ...]], domain: Tuple[int, ...]):
        frozen_hash = shash(origin, domain)
        # check if same sdfg already cached on disk
        path = self.builder.module_path
        basename = os.path.splitext(path)[0]
        path = basename + "_" + str(frozen_hash) + ".sdfg"
        if path not in self._loaded_sdfgs:
            try:
                sdfg = dace.SDFG.from_file(path)
            except FileNotFoundError:
                # otherwise, wrap and save sdfg from scratch
                inner_sdfg = self.unexpanded_sdfg()

                sdfg = freeze_origin_domain_sdfg(
                    inner_sdfg,
                    arg_names=[arg.name for arg in self.builder.gtir.api_signature],
                    field_info=make_args_data_from_gtir(self.builder.gtir_pipeline).field_info,
                    origin=origin,
                    domain=domain,
                )
                self._save_sdfg(sdfg, path)
            self._loaded_sdfgs[path] = sdfg
        return self._loaded_sdfgs[path]

    def frozen_sdfg(self, *, origin: Dict[str, Tuple[int, ...]], domain: Tuple[int, ...]):
        return copy.deepcopy(self._frozen_sdfg(origin=origin, domain=domain))


class DaCeExtGenerator(BackendCodegen):
    def __init__(self, class_name, module_name, backend):
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self, stencil_ir: gtir.Stencil) -> Dict[str, Dict[str, str]]:

        manager = SDFGManager(self.backend.builder)
        sdfg = manager.expanded_sdfg()

        sources: Dict[str, Dict[str, str]]
        implementation = DaCeComputationCodegen.apply(stencil_ir, self.backend.builder, sdfg)

        bindings = DaCeBindingsCodegen.apply(
            stencil_ir, sdfg, module_name=self.module_name, backend=self.backend
        )

        bindings_ext = "cu" if self.backend.storage_info["device"] == "gpu" else "cpp"
        sources = {
            "computation": {"computation.hpp": implementation},
            "bindings": {f"bindings.{bindings_ext}": bindings},
        }
        return sources


class DaCeComputationCodegen:

    template = as_mako(
        """
        auto ${name}(const std::array<gt::uint_t, 3>& domain) {
            return [domain](${",".join(functor_args)}) {
                const int __I = domain[0];
                const int __J = domain[1];
                const int __K = domain[2];
                ${name}_t dace_handle;
                auto allocator = gt::sid::cached_allocator(&${allocator}<char[]>);
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
                name=f"__{array_sdfg.sdfg_id}_{name}",
                dtype=array.dtype.ctype,
                size=array.total_size,
            )
            for array_sdfg, name, array in sdfg.arrays_recursive()
            if array.transient and array.lifetime == dace.AllocationLifetime.Persistent
        ]

    @staticmethod
    def _postprocess_dace_code(code_objects, is_gpu, builder):
        lines = code_objects[[co.title for co in code_objects].index("Frame")].clean_code.split(
            "\n"
        )

        if is_gpu:
            regex = re.compile("struct [a-zA-Z_][a-zA-Z0-9_]*_t {")
            for i, line in enumerate(lines):
                if regex.match(line.strip()):
                    j = i + 1
                    while "};" not in lines[j].strip():
                        j += 1
                    lines = lines[0:i] + lines[j + 1 :]
                    break
            for i, line in enumerate(lines):
                if "#include <dace/dace.h>" in line:
                    cuda_code = [co.clean_code for co in code_objects if co.title == "CUDA"][0]
                    lines = lines[0:i] + cuda_code.split("\n") + lines[i + 1 :]
                    break

        def keep_line(line):
            line = line.strip()
            if line == '#include "../../include/hash.h"':
                return False
            if line.startswith("DACE_EXPORTED") and line.endswith(");"):
                return False
            if line == "#include <cuda_runtime.h>":
                return False
            return True

        lines = filter(keep_line, lines)
        generated_code = "\n".join(lines)
        if builder.options.format_source:
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return generated_code

    @classmethod
    def apply(cls, stencil_ir: gtir.Stencil, builder: "StencilBuilder", sdfg: dace.SDFG):
        self = cls()
        with dace.config.temporary_config():
            dace.config.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)
            dace.config.Config.set("compiler", "cpu", "openmp_sections", value=False)
            code_objects = sdfg.generate_code()
        is_gpu = "CUDA" in {co.title for co in code_objects}

        computations = cls._postprocess_dace_code(code_objects, is_gpu, builder)

        interface = cls.template.definition.render(
            name=sdfg.name,
            dace_args=self.generate_dace_args(stencil_ir, sdfg),
            functor_args=self.generate_functor_args(sdfg),
            tmp_allocs=self.generate_tmp_allocs(sdfg),
            allocator="gt::cuda_util::cuda_malloc" if is_gpu else "std::make_unique",
        )
        generated_code = textwrap.dedent(
            f"""#include <gridtools/sid/sid_shift_origin.hpp>
                             #include <gridtools/sid/allocator.hpp>
                             #include <gridtools/stencil/cartesian.hpp>
                             {"#include <gridtools/common/cuda_util.hpp>" if is_gpu else ""}
                             namespace gt = gridtools;
                             {computations}

                             {interface}
                             """
        )
        if builder.options.format_source:
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code

    def generate_dace_args(self, stencil_ir: gtir.Stencil, sdfg: dace.SDFG) -> List[str]:
        oir = GTIRToOIR().visit(stencil_ir)
        field_extents = compute_fields_extents(oir, add_k=True)

        k_origins = {
            field_name: max(boundary[0], 0)
            for field_name, boundary in compute_k_boundary(stencil_ir).items()
        }
        offset_dict: Dict[str, Tuple[int, int, int]] = {
            k: (max(-v[0][0], 0), max(-v[1][0], 0), k_origins[k] if k in k_origins else 0)
            for k, v in field_extents.items()
        }

        symbols = {f"__{var}": f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            # transients are set to expressions based on their shape in _specialize_transient_strides
            if array.transient:
                continue

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

    def generate_entry_params(self, stencil_ir: gtir.Stencil, sdfg: dace.SDFG) -> List[str]:
        res: Dict[str, str] = {}
        import dace.data

        for name in sdfg.signature_arglist(with_types=False, for_call=True):
            if name in sdfg.arrays:
                data = sdfg.arrays[name]
                assert isinstance(data, dace.data.Array)
                res[name] = "py::buffer {name}, std::array<gt::int_t,{ndim}> {name}_origin".format(
                    name=name,
                    ndim=len(data.shape),
                )
            elif name in sdfg.symbols and not name.startswith("__"):
                assert name in sdfg.symbols
                res[name] = "{dtype} {name}".format(dtype=sdfg.symbols[name].ctype, name=name)
        return list(res[node.name] for node in stencil_ir.params if node.name in res)

    def generate_sid_params(self, sdfg: dace.SDFG) -> List[str]:
        res: List[str] = []
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
                check_layout=False,
                backend=self.backend,
            )

            res.append(sid_def)
        # pass scalar parameters as variables
        for name in (n for n in sdfg.symbols.keys() if not n.startswith("__")):
            res.append(name)
        return res

    def generate_sdfg_bindings(self, stencil_ir: gtir.Stencil, sdfg, module_name) -> str:

        return self.mako_template.render_values(
            name=sdfg.name,
            module_name=module_name,
            entry_params=self.generate_entry_params(stencil_ir, sdfg),
            sid_params=self.generate_sid_params(sdfg),
        )

    @classmethod
    def apply(cls, stencil_ir: gtir.Stencil, sdfg: dace.SDFG, module_name: str, *, backend) -> str:
        generated_code = cls(backend).generate_sdfg_bindings(
            stencil_ir, sdfg, module_name=module_name
        )
        if backend.builder.options.format_source:
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return generated_code


class DaCePyExtModuleGenerator(PyExtModuleGenerator):
    def generate_imports(self):
        return "\n".join(
            [
                *super().generate_imports().splitlines(),
                "import dace",
                "import copy",
                "from gt4py.backend.dace_stencil_object import DaCeStencilObject",
            ]
        )

    def generate_base_class_name(self):
        return "DaCeStencilObject"

    def generate_class_members(self):
        res = super().generate_class_members()
        filepath = self.builder.module_path.joinpath(
            os.path.dirname(self.builder.module_path), self.builder.module_name + ".sdfg"
        )
        res += f'\nSDFG_PATH = "{filepath}"\n'
        return res


class DaCeCUDAPyExtModuleGenerator(DaCePyExtModuleGenerator, GTCUDAPyModuleGenerator):
    pass


class BaseDaceBackend(BaseGTBackend, CLIBackendMixin):

    GT_BACKEND_T = "dace"
    PYEXT_GENERATOR_CLASS = DaCeExtGenerator  # type: ignore

    options = BaseGTBackend.GT_BACKEND_OPTS

    def generate(self) -> Type["StencilObject"]:
        self.check_options(self.builder.options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not gt_src_manager.has_gt_sources() and not gt_src_manager.install_gt_sources():
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


@register
class DaceCPUBackend(BaseDaceBackend):

    name = "dace:cpu"
    languages = {"computation": "c++", "bindings": ["python"]}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": layout_maker_factory((1, 0, 2)),
        "is_compatible_layout": lambda x: True,
        "is_compatible_type": lambda x: isinstance(x, np.ndarray),
    }
    MODULE_GENERATOR_CLASS = DaCePyExtModuleGenerator

    options = BaseGTBackend.GT_BACKEND_OPTS

    def generate_extension(self) -> Tuple[str, str]:
        return self.make_extension(stencil_ir=self.builder.gtir, uses_cuda=False)


@register
class DaceGPUBackend(BaseDaceBackend):
    """DaCe python backend using gtc."""

    name = "dace:gpu"
    languages = {"computation": "cuda", "bindings": ["python"]}
    storage_info = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": layout_maker_factory((2, 1, 0)),
        "is_compatible_layout": lambda x: True,
        "is_compatible_type": cuda_is_compatible_type,
    }
    MODULE_GENERATOR_CLASS = DaCeCUDAPyExtModuleGenerator
    options = {
        **BaseGTBackend.GT_BACKEND_OPTS,
        "device_sync": {"versioning": True, "type": bool},
    }

    def generate_extension(self) -> Tuple[str, str]:
        return self.make_extension(stencil_ir=self.builder.gtir, uses_cuda=True)
