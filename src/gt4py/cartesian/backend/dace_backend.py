# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import pathlib
import re
from typing import TYPE_CHECKING, ClassVar

from dace import SDFG, Memlet, SDFGState, config, data, dtypes, nodes, subsets, symbolic
from dace.codegen import codeobject
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.utils import inline_sdfgs
from dace.transformation.passes import SimplifyPass

from gt4py._core import definitions as core_defs
from gt4py.cartesian import config as gt_config, definitions
from gt4py.cartesian.backend.base import register
from gt4py.cartesian.backend.gtc_common import (
    BackendCodegen,
    BaseGTBackend,
    CUDAPyExtModuleGenerator,
    GTBackendOptions,
    PyExtModuleGenerator,
    bindings_main_template,
    pybuffer_to_sid,
)
from gt4py.cartesian.backend.module_generator import make_args_data_from_gtir
from gt4py.cartesian.gtc import gtir
from gt4py.cartesian.gtc.dace import passes
from gt4py.cartesian.gtc.dace.oir_to_treeir import OIRToTreeIR
from gt4py.cartesian.gtc.dace.treeir_to_stree import TreeIRToScheduleTree
from gt4py.cartesian.gtc.dace.utils import array_dimensions, replace_strides
from gt4py.cartesian.gtc.definitions import CartesianSpace
from gt4py.cartesian.gtc.gtir_to_oir import GTIRToOIR
from gt4py.cartesian.gtc.passes.gtir_k_boundary import compute_k_boundary
from gt4py.cartesian.gtc.passes.oir_optimizations import caches
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import compute_fields_extents
from gt4py.cartesian.gtc.passes.oir_pipeline import DefaultPipeline
from gt4py.cartesian.utils import shash
from gt4py.eve import codegen
from gt4py.eve.codegen import MakoTemplate as as_mako
from gt4py.storage.cartesian import layout


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_builder import StencilBuilder
    from gt4py.cartesian.stencil_object import StencilObject


def _specialize_transient_strides(
    sdfg: SDFG, layout_info: layout.LayoutInfo, replacement_dictionary: dict[str, str] | None = None
) -> None:
    # Find transients in this SDFG to specialize.
    stride_replacements = replace_strides(
        [
            array
            for array in sdfg.arrays.values()
            if isinstance(array, data.Array) and array.transient
        ],
        layout_info,
    )

    # In case of nested SDFGs (see below), merge with replacement dict that was passed down.
    # Dev note: We shouldn't use mutable data structures as argument defaults.
    replacement_dictionary = {} if replacement_dictionary is None else replacement_dictionary
    replacement_dictionary.update(stride_replacements)

    # Replace in this SDFG
    sdfg.replace_dict(replacement_dictionary)
    for state in sdfg.nodes():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                # Recursively replace strides in nested SDFGs
                _specialize_transient_strides(node.sdfg, layout_info, replacement_dictionary)
    for k in replacement_dictionary.keys():
        if k in sdfg.symbols:
            sdfg.remove_symbol(k)


def _sdfg_add_arrays_and_edges(
    field_info: dict[str, definitions.FieldInfo],
    wrapper_sdfg: SDFG,
    state: SDFGState,
    inner_sdfg: SDFG,
    nsdfg: nodes.NestedSDFG,
    inputs: set[str] | dict[str, dtypes.typeclass],
    outputs: set[str] | dict[str, dtypes.typeclass],
    origins: dict[str, tuple[int, ...]],
) -> None:
    for name, array in inner_sdfg.arrays.items():
        if array.transient:
            continue

        if isinstance(array, data.Array):
            axes = field_info[name].axes

            shape = [f"__{name}_{axis}_size" for axis in axes] + [
                d for d in field_info[name].data_dims
            ]

            wrapper_sdfg.add_array(
                name,
                dtype=array.dtype,
                strides=array.strides,
                shape=shape,
                storage=array.storage,
            )

            # Calculate memlet ranges taking the origin into account
            ranges = []
            origin = origins.get(name, origins.get("_all_", None))

            if origin is None:
                raise ValueError("Freeze origin/domain: Unspecified origin for field `{name}`.")

            # Read boundaries for axis-bound fields
            index = 0
            for cartesian_index, axis in enumerate(CartesianSpace.names):
                if axis not in axes:
                    continue
                o = origin[index]
                e = field_info[name].boundary.lower_indices[cartesian_index]
                s = inner_sdfg.arrays[name].shape[index]
                ranges.append(
                    # s - 1 because ranges are inclusive
                    (o - max(0, e), o - max(0, e) + s - 1, 1)
                )
                index += 1

            # Add data dimensions to the range
            # NOTE: origin and boundary of data dimensions are always 0
            ranges += [
                (0, d - 1, 1)  # d - 1 because ranges are inclusive
                for d in field_info[name].data_dims
            ]

            if name in inputs:
                state.add_edge(
                    state.add_read(name),
                    None,
                    nsdfg,
                    name,
                    Memlet(name, subset=subsets.Range(ranges)),
                )
            if name in outputs:
                state.add_edge(
                    nsdfg,
                    name,
                    state.add_write(name),
                    None,
                    Memlet(name, subset=subsets.Range(ranges)),
                )
        elif isinstance(array, data.Scalar):
            wrapper_sdfg.add_scalar(
                name,
                dtype=array.dtype,
                storage=array.storage,
                lifetime=array.lifetime,
            )
            if name in inputs:
                state.add_edge(
                    state.add_read(name),
                    None,
                    nsdfg,
                    name,
                    Memlet(name),
                )
            if name in outputs:
                state.add_edge(
                    nsdfg,
                    name,
                    state.add_write(name),
                    None,
                    Memlet(name),
                )


def _sdfg_specialize_symbols(wrapper_sdfg: SDFG, domain: tuple[int, ...]) -> None:
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
            sym = symbolic.pystr_to_symbolic(val)
            for fsym in sym.free_symbols:
                if sdfg.parent_nsdfg_node is not None:
                    sdfg.parent_nsdfg_node.symbol_mapping[str(fsym)] = fsym
                if str(fsym) not in sdfg.symbols:
                    if str(fsym) in sdfg.parent_sdfg.symbols:
                        sdfg.add_symbol(str(fsym), stype=sdfg.parent_sdfg.symbols[str(fsym)])
                    else:
                        sdfg.add_symbol(str(fsym), stype=dtypes.int32)


def freeze_origin_domain_sdfg(
    inner_sdfg_unfrozen: SDFG,
    arg_names: list[str],
    field_info: dict[str, definitions.FieldInfo],
    *,
    layout_info: layout.LayoutInfo,
    origin: dict[str, tuple[int, ...]],
    domain: tuple[int, ...],
) -> SDFG:
    """Create a new SDFG by wrapping a _copy_ of the original SDFG and freezing it's
    origin and domain.

    This wrapping is required because we do not expect any of the inner_sdfg bounds to
    have been specialized, e.g. we expect "__I/J/K" symbols to still be present. We wrap
    the call and specialize at top level, which will then be passed as a symbol to the
    inner sdfg.

    Once we move specialization of array & maps bounds upstream, this will become moot
    and can be removed, see https://github.com/GridTools/gt4py/issues/2082.

    Dev note: we need to wrap a copy to make sure we can use caching with no side effects
    in other parts of the SDFG making pipeline.

    Args:
        inner_sdfg_unfrozen: SDFG with cartesian bounds as symbols
        arg_names: names of arguments to freeze
        field_info: full info stack on arguments
        origin: tuple of offset into the memory
        domain: tuple of size for the memory written by the stencil
    """
    inner_sdfg = copy.deepcopy(inner_sdfg_unfrozen)

    wrapper_sdfg = SDFG("frozen_" + inner_sdfg.name)
    state = wrapper_sdfg.add_state("frozen_" + inner_sdfg.name + "_state")

    inputs = set()
    outputs = set()
    for inner_state in inner_sdfg.nodes():
        for node in inner_state.nodes():
            if not isinstance(node, nodes.AccessNode) or inner_sdfg.arrays[node.data].transient:
                continue
            if node.has_reads(inner_state):
                inputs.add(node.data)
            if node.has_writes(inner_state):
                outputs.add(node.data)

    nsdfg = state.add_nested_sdfg(inner_sdfg, None, inputs, outputs)

    _sdfg_add_arrays_and_edges(
        field_info, wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs, origin
    )

    # in special case of empty domain, remove entire SDFG.
    if any(d == 0 for d in domain):
        states = wrapper_sdfg.states()
        assert len(states) == 1
        for node in states[0].nodes():
            state.remove_node(node)

    # make sure that symbols are passed through to inner sdfg
    for symbol in nsdfg.sdfg.free_symbols:
        if symbol not in wrapper_sdfg.symbols:
            wrapper_sdfg.add_symbol(symbol, nsdfg.sdfg.symbols[symbol])

    # Try to inline wrapped SDFG before symbols are specialized to avoid extra views
    inline_sdfgs(wrapper_sdfg)

    _sdfg_specialize_symbols(wrapper_sdfg, domain)
    _specialize_transient_strides(wrapper_sdfg, layout_info)

    for _, _, array in wrapper_sdfg.arrays_recursive():
        if array.transient:
            array.lifetime = dtypes.AllocationLifetime.SDFG

    wrapper_sdfg.arg_names = arg_names

    return wrapper_sdfg


class SDFGManager:
    # Cache loaded SDFGs across all instances (unless caching strategy is "nocaching")
    _loaded_sdfgs: ClassVar[dict[str | pathlib.Path, SDFG]] = dict()

    def __init__(self, builder: StencilBuilder, debug_stree: bool = False) -> None:
        """
        Initializes the SDFGManager.

        Args:
          builder: The StencilBuilder instance, used for build options and caching strategy.
          debug_stree: If true, saves a string representation of the schedule tree next to the cached SDFG.
        """
        self.builder = builder
        self.debug_stree = debug_stree

    def schedule_tree(self) -> tn.ScheduleTreeRoot:
        """
        Schedule tree representation of the gtir (taken from the builder).

        This function is a three-step process:

        oir = gtir_to_oir(self.builder.gtir)
        tree_ir = oir_to_tree_ir(oir)
        schedule_tree = tree_ir_to_schedule_tree(tree_ir)
        """

        oir = GTIRToOIR().visit(self.builder.gtir)

        # Deactivate caches. We need to extend the skip list in case users have
        # specified skip as well AND we need to copy in order to not trash the
        # cache hash!
        oir_pipeline: DefaultPipeline = self.builder.options.backend_opts.get(
            "oir_pipeline", DefaultPipeline()
        )
        oir_pipeline = copy.deepcopy(oir_pipeline)
        oir_pipeline.skip.extend(
            [
                caches.IJCacheDetection,
                caches.KCacheDetection,
                caches.PruneKCacheFills,
                caches.PruneKCacheFlushes,
            ]
        )
        oir = oir_pipeline.run(oir)

        tir = OIRToTreeIR(self.builder).visit(oir)

        return TreeIRToScheduleTree().visit(tir)

    @staticmethod
    def _strip_history(sdfg: SDFG) -> None:
        # strip history from SDFG for faster save/load
        for tmp_sdfg in sdfg.all_sdfgs_recursive():
            tmp_sdfg.transformation_hist = []
            tmp_sdfg.orig_sdfg = None

    @staticmethod
    def _save_sdfg(sdfg: SDFG, path: pathlib.Path, validate: bool = False) -> None:
        if validate:
            sdfg.validate()
        SDFGManager._strip_history(sdfg)
        sdfg.save(str(path))

    def sdfg_via_schedule_tree(self, *, validate: bool = True, simplify: bool = True) -> SDFG:
        """Lower OIR into an SDFG via Schedule Tree transpile first.

        Cache the SDFG into the manager for re-use, unless the builder has a no-caching policy.

        Args:
            validate: Validate resulting SDFG
            simplify: Simplify resulting SDFG
        """
        filename = f"{self.builder.module_name}.sdfg"
        path = (
            pathlib.Path(os.path.relpath(self.builder.module_path.parent, pathlib.Path.cwd()))
            / filename
        )

        do_cache = self.builder.caching.name != "nocaching"
        if do_cache:
            if path in SDFGManager._loaded_sdfgs:
                return SDFGManager._loaded_sdfgs[path]
            if path.exists():
                sdfg = SDFG.from_file(path)
                SDFGManager._loaded_sdfgs[path] = sdfg
                return sdfg

        # Create SDFG
        stree = self.schedule_tree()

        if self.builder.backend.name.endswith("_kfirst"):
            # re-order loops to match loops with memory layout
            flipper = passes.PushVerticalMapDown()
            flipper.visit(stree)

        sdfg = stree.as_sdfg(
            validate=validate,
            simplify=simplify,
            skip={"ScalarToSymbolPromotion"},
        )

        if do_cache:
            self._save_sdfg(sdfg, path)
            SDFGManager._loaded_sdfgs[path] = sdfg

            if self.debug_stree:
                stree_path = path.with_suffix(".stree.txt")
                with open(stree_path, "w+") as file:
                    file.write(stree.as_string(-1))

        return sdfg

    def _frozen_sdfg(self, *, origin: dict[str, tuple[int, ...]], domain: tuple[int, ...]) -> SDFG:
        basename = self.builder.module_path.with_suffix("")
        path = pathlib.Path(f"{basename}_{shash(origin, domain)}.sdfg")

        # check if the same sdfg is already loaded
        do_cache = self.builder.caching.name != "nocache"
        if do_cache:
            if path in SDFGManager._loaded_sdfgs:
                return SDFGManager._loaded_sdfgs[path]
            if path.exists():
                sdfg = SDFG.from_file(path)
                SDFGManager._loaded_sdfgs[path] = sdfg
                return sdfg

        # Otherwise, wrap and save sdfg from scratch
        sdfg = self.sdfg_via_schedule_tree()
        frozen_sdfg = freeze_origin_domain_sdfg(
            sdfg,
            arg_names=[arg.name for arg in self.builder.gtir.api_signature],
            field_info=make_args_data_from_gtir(self.builder.gtir_pipeline).field_info,
            layout_info=self.builder.backend.storage_info,
            origin=origin,
            domain=domain,
        )

        if do_cache:
            SDFGManager._loaded_sdfgs[path] = frozen_sdfg
            self._save_sdfg(frozen_sdfg, path)

        return frozen_sdfg

    def frozen_sdfg(self, *, origin: dict[str, tuple[int, ...]], domain: tuple[int, ...]) -> SDFG:
        return copy.deepcopy(self._frozen_sdfg(origin=origin, domain=domain))


class DaCeExtGenerator(BackendCodegen):
    def __init__(self, class_name: str, module_name: str, backend: BaseDaceBackend) -> None:
        self.class_name = class_name
        self.module_name = module_name
        self.backend = backend

    def __call__(self) -> dict[str, dict[str, str]]:
        manager = SDFGManager(self.backend.builder)

        sdfg = manager.sdfg_via_schedule_tree()
        _specialize_transient_strides(
            sdfg,
            self.backend.storage_info,
        )
        SimplifyPass(validate=True, skip={"ScalarToSymbolPromotion"}).apply_pass(sdfg, {})

        # NOTE
        # The glue code in DaCeComputationCodegen.apply() (just below) will define all the
        # symbols. Our job creating the sdfg/stree is to make sure we use the same symbols
        # and to be sure that these symbols are added as dace symbols.

        implementation = DaCeComputationCodegen.apply(self.backend.builder, sdfg)

        bindings = DaCeBindingsCodegen.apply(sdfg, self.module_name, backend=self.backend)

        bindings_ext = "cu" if self.backend.storage_info["device"] == "gpu" else "cpp"
        return {
            "computation": {"computation.hpp": implementation},
            "bindings": {f"bindings.{bindings_ext}": bindings},
        }


class DaCeComputationCodegen:
    template = as_mako(
        """\
auto ${name}(const std::array<gt::uint_t, 3>& domain) {
    return [domain](${",".join(functor_args)}) {
        const int __I = domain[0];
        const int __J = domain[1];
        const int __K = domain[2];
        ${name}${state_suffix} dace_handle;
        ${backend_specifics}
        auto allocator = gt::sid::cached_allocator(&${allocator}<char[]>);
        ${"\\n".join(tmp_allocs)}
        __program_${name}(${",".join(["&dace_handle", *dace_args])});
    };
}
"""
    )

    def generate_tmp_allocs(self, sdfg: SDFG) -> list[str]:
        global_fmt = (
            "__{sdfg_id}_{name} = allocate(allocator, gt::meta::lazy::id<{dtype}>(), {size})();"
        )
        threadlocal_fmt = (
            "__{sdfg_id}_{name} = __full__{sdfg_id}_{name} + omp_get_thread_num() * ({local_size});"
        )
        res = []
        for array_sdfg, name, array in sdfg.arrays_recursive():
            if array.transient and array.lifetime == dtypes.AllocationLifetime.Persistent:
                if array.storage != dtypes.StorageType.CPU_ThreadLocal:
                    fmt = "dace_handle." + global_fmt
                    res.append(
                        fmt.format(
                            name=name,
                            sdfg_id=array_sdfg.sdfg_id,
                            dtype=array.dtype.ctype,
                            size=array.total_size,
                        )
                    )
                else:
                    fmts = [
                        "{dtype} *__full" + global_fmt,
                        "#pragma omp parallel",
                        "{{",
                        threadlocal_fmt,
                        "}}",
                    ]
                    res.extend(
                        [
                            fmt.format(
                                name=name,
                                sdfg_id=array_sdfg.sdfg_id,
                                dtype=array.dtype.ctype,
                                size=f"omp_max_threads * ({array.total_size})",
                                local_size=array.total_size,
                            )
                            for fmt in fmts
                        ]
                    )
        return res

    @staticmethod
    def _postprocess_dace_code(code_objects: list[codeobject.CodeObject], is_gpu: bool) -> str:
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
                    cuda_code = next(co.clean_code for co in code_objects if co.title == "CUDA")
                    lines = lines[0:i] + cuda_code.split("\n") + lines[i + 1 :]
                    break

        def keep_line(line: str) -> bool:
            line = line.strip()
            if line == '#include "../../include/hash.h"':
                return False
            if line.startswith("DACE_EXPORTED") and line.endswith(");"):
                return False
            if line == "#include <cuda_runtime.h>":
                return False
            return True

        return "\n".join(filter(keep_line, lines))

    @classmethod
    def apply(cls, builder: StencilBuilder, sdfg: SDFG) -> str:
        self = cls()
        with config.temporary_config():
            # To prevent conflict with 3rd party usage of DaCe config always make sure that any
            #  changes be under the temporary_config manager
            if core_defs.CUPY_DEVICE_TYPE == core_defs.DeviceType.ROCM:
                config.Config.set("compiler", "cuda", "backend", value="hip")
            config.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)
            config.Config.set(
                "compiler", "cuda", "default_block_size", value=gt_config.DACE_DEFAULT_BLOCK_SIZE
            )
            config.Config.set("compiler", "cpu", "openmp_sections", value=False)
            code_objects = sdfg.generate_code()
        is_gpu = "CUDA" in {co.title for co in code_objects}

        computations = cls._postprocess_dace_code(code_objects, is_gpu)
        if not is_gpu and any(
            array.transient and array.lifetime == dtypes.AllocationLifetime.Persistent
            for *_, array in sdfg.arrays_recursive()
        ):
            omp_threads = "int omp_max_threads = omp_get_max_threads();"
            omp_header = "#include <omp.h>"
        else:
            omp_threads = ""
            omp_header = ""

        interface = cls.template.definition.render(
            name=sdfg.name,
            backend_specifics=omp_threads,
            dace_args=self.generate_dace_args(builder.gtir, sdfg),
            functor_args=self.generate_functor_args(sdfg),
            tmp_allocs=self.generate_tmp_allocs(sdfg),
            allocator="gt::cuda_util::cuda_malloc" if is_gpu else "std::make_unique",
            state_suffix=config.Config.get("compiler.codegen_state_struct_suffix"),
        )
        generated_code = f"""\
#include <gridtools/sid/sid_shift_origin.hpp>
#include <gridtools/sid/allocator.hpp>
#include <gridtools/stencil/cartesian.hpp>
{"#include <gridtools/common/cuda_util.hpp>" if is_gpu else omp_header}
namespace gt = gridtools;

{computations}

{interface}
"""

        if builder.options.format_source:
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")

        return generated_code

    def generate_dace_args(self, stencil_ir: gtir.Stencil, sdfg: SDFG) -> list[str]:
        oir = GTIRToOIR().visit(stencil_ir)
        field_extents = compute_fields_extents(oir, add_k=True)

        k_origins = {
            field_name: max(boundary[0], 0)
            for field_name, boundary in compute_k_boundary(stencil_ir).items()
        }
        offset_dict: dict[str, tuple[int, int, int]] = {
            k: (max(-v[0][0], 0), max(-v[1][0], 0), k_origins[k] if k in k_origins else 0)
            for k, v in field_extents.items()
        }

        symbols = {f"__{var}": f"__{var}" for var in "IJK"}
        for name, array in sdfg.arrays.items():
            # transients are set to expressions based on their shape in _specialize_transient_strides
            if array.transient:
                continue

            if isinstance(array, data.Scalar):
                # will be passed by name (as variable) by the catch all below
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
                    symbolic.pystr_to_symbolic(f"__{var}") in s.free_symbols
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

    def generate_functor_args(self, sdfg: SDFG) -> list[str]:
        arguments = []
        for name, array in sdfg.arrays.items():
            if array.transient:
                continue
            if isinstance(array, data.Scalar):
                arguments.append(f"auto {name}")
                continue
            if isinstance(array, data.Array):
                arguments.append(f"auto && __{name}_sid")
                continue
            raise NotImplementedError(f"generate_functor_args(): unexpected type {type(array)}")
        for name, dtype in ((n, d) for n, d in sdfg.symbols.items() if not n.startswith("__")):
            arguments.append(dtype.as_arg(name))
        return arguments


class DaCeBindingsCodegen:
    def __init__(self, backend: BaseDaceBackend) -> None:
        self.backend = backend
        self._unique_index: int = 0

    def unique_index(self) -> int:
        self._unique_index += 1
        return self._unique_index

    mako_template = bindings_main_template()

    def generate_entry_params(self, sdfg: SDFG) -> list[str]:
        res: dict[str, str] = {}

        for name in sdfg.signature_arglist(with_types=False, for_call=True):
            if name in sdfg.arrays:
                container = sdfg.arrays[name]
                if isinstance(container, data.Scalar):
                    res[name] = f"{container.ctype} {name}"
                elif isinstance(container, data.Array):
                    res[name] = (
                        "py::{pybind_type} {name}, std::array<gt::int_t,{ndim}> {name}_origin".format(
                            pybind_type=(
                                "object"
                                if self.backend.storage_info["device"] == "gpu"
                                else "buffer"
                            ),
                            name=name,
                            ndim=len(container.shape),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"generate_entry_params(): unexpected type {type(container)}"
                    )
            elif name in sdfg.symbols and not name.startswith("__"):
                res[name] = f"{sdfg.symbols[name].ctype} {name}"
        return list(res[node.name] for node in self.backend.builder.gtir.params if node.name in res)

    def generate_sid_params(self, sdfg: SDFG) -> list[str]:
        res: list[str] = []

        for name, array in sdfg.arrays.items():
            if array.transient:
                continue

            if isinstance(array, data.Scalar):
                res.append(name)
                continue

            if not isinstance(array, data.Array):
                raise NotImplementedError(f"generate_sid_params(): unexpected type {type(array)}")

            domain_dim_flags = tuple(array_dimensions(array))
            if len(domain_dim_flags) != 3:
                raise RuntimeError("Expected 3 cartesian array dimensions. Codegen error.")
            data_ndim = len(array.shape) - sum(domain_dim_flags)
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

    def generate_sdfg_bindings(self, sdfg: SDFG, module_name: str) -> str:
        return self.mako_template.render_values(
            name=sdfg.name,
            module_name=module_name,
            entry_params=self.generate_entry_params(sdfg),
            sid_params=self.generate_sid_params(sdfg),
        )

    @classmethod
    def apply(cls, sdfg: SDFG, module_name: str, *, backend: BaseDaceBackend) -> str:
        generated_code = cls(backend).generate_sdfg_bindings(sdfg, module_name)
        if backend.builder.options.format_source:
            generated_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return generated_code


class DaCePyExtModuleGenerator(PyExtModuleGenerator):
    def __init__(self, builder: StencilBuilder) -> None:
        super().__init__(builder)

    def generate_imports(self) -> str:
        return "\n".join(
            [
                *super().generate_imports().splitlines(),
                "import dace",
                "import copy",
                "from gt4py.cartesian.backend.dace_stencil_object import DaCeStencilObject",
            ]
        )

    def generate_base_class_name(self) -> str:
        return "DaCeStencilObject"

    def generate_class_members(self) -> str:
        res = super().generate_class_members()
        filepath = self.builder.module_path.joinpath(
            os.path.dirname(self.builder.module_path), self.builder.module_name + ".sdfg"
        )
        res += f'\nSDFG_PATH = "{filepath}"\n'
        return res


class DaCeCUDAPyExtModuleGenerator(DaCePyExtModuleGenerator, CUDAPyExtModuleGenerator):
    pass


class BaseDaceBackend(BaseGTBackend):
    GT_BACKEND_T = "dace"
    PYEXT_GENERATOR_CLASS = DaCeExtGenerator

    def generate(self) -> type[StencilObject]:
        self.check_options(self.builder.options)

        # TODO(havogt) add bypass if computation has no effect
        self.generate_extension()

        # Generate and return the Python wrapper class
        return self.make_module()


@register
class DaceCPUBackend(BaseDaceBackend):
    name = "dace:cpu"
    languages: ClassVar[dict] = {"computation": "c++", "bindings": ["python"]}
    storage_info: ClassVar[layout.LayoutInfo] = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": layout.layout_maker_factory((1, 2, 0)),
        "is_optimal_layout": layout.layout_checker_factory(layout.layout_maker_factory((1, 2, 0))),
    }
    MODULE_GENERATOR_CLASS = DaCePyExtModuleGenerator

    options = BaseGTBackend.GT_BACKEND_OPTS

    def generate_extension(self) -> None:
        return self.make_extension(uses_cuda=False)


@register
class DaceCPUKFirstBackend(BaseDaceBackend):
    name = "dace:cpu_kfirst"
    languages: ClassVar[dict] = {"computation": "c++", "bindings": ["python"]}
    storage_info: ClassVar[layout.LayoutInfo] = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": layout.layout_maker_factory((0, 1, 2)),
        "is_optimal_layout": layout.layout_checker_factory(layout.layout_maker_factory((0, 1, 2))),
    }
    MODULE_GENERATOR_CLASS = DaCePyExtModuleGenerator

    options = BaseGTBackend.GT_BACKEND_OPTS

    def generate_extension(self) -> None:
        return self.make_extension(uses_cuda=False)


@register
class DaceGPUBackend(BaseDaceBackend):
    """DaCe python backend using gt4py.cartesian.gtc."""

    name = "dace:gpu"
    languages: ClassVar[dict] = {"computation": "cuda", "bindings": ["python"]}
    storage_info: ClassVar[layout.LayoutInfo] = {
        "alignment": 32,
        "device": "gpu",
        "layout_map": layout.layout_maker_factory((2, 1, 0)),
        "is_optimal_layout": layout.layout_checker_factory(layout.layout_maker_factory((2, 1, 0))),
    }
    MODULE_GENERATOR_CLASS = DaCeCUDAPyExtModuleGenerator
    options: ClassVar[GTBackendOptions] = {
        **BaseGTBackend.GT_BACKEND_OPTS,
        "device_sync": {"versioning": True, "type": bool},
    }

    def generate_extension(self) -> None:
        return self.make_extension(uses_cuda=True)
