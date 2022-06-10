# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import copy
import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import dace
import dace.data
import dace.frontend.python.common
from dace.frontend.python.common import SDFGClosure, SDFGConvertible
from dace.sdfg.utils import inline_sdfgs

import gt4py.backend as gt_backend
from gt4py.definitions import AccessKind
from gt4py.stencil_object import FrozenStencil, StencilObject
from gt4py.utils import shash


@dataclass(frozen=True)
class DaCeFrozenStencil(FrozenStencil, SDFGConvertible):

    stencil_object: "DaCeStencilObject"
    origin: Dict[str, Tuple[int, ...]]
    domain: Tuple[int, ...]
    sdfg: dace.SDFG

    def _add_optionals(self, sdfg, **kwargs):
        for name, info in self.stencil_object.field_info.items():
            if info.access == AccessKind.NONE and name in kwargs:
                outer_array = kwargs[name]
                sdfg.add_array(
                    name,
                    shape=outer_array.shape,
                    dtype=outer_array.dtype,
                    strides=outer_array.strides,
                )

        for name, info in self.stencil_object.parameter_info.items():
            if info.access == AccessKind.NONE and name in kwargs:
                if isinstance(kwargs[name], dace.data.Scalar):
                    sdfg.add_symbol(name, stype=kwargs[name].dtype)
                else:
                    sdfg.add_symbol(name, stype=dace.typeclass(type(kwargs[name])))
        return sdfg

    def __sdfg__(self, **kwargs):
        return self._add_optionals(copy.deepcopy(self.sdfg), **kwargs)

    def __sdfg_signature__(self):
        return self.stencil_object.__sdfg_signature__()

    def __sdfg_closure__(self, *args, **kwargs):
        return {}

    def closure_resolver(self, constant_args, given_args, parent_closure=None):
        return SDFGClosure()


class DaCeStencilObject(StencilObject, SDFGConvertible):

    _sdfg = None
    SDFG_PATH: str

    def __new__(cls, *args, **kwargs):
        res = super().__new__(cls, *args, **kwargs)
        if getattr(cls, "_frozen_cache", None) is None:
            cls._frozen_cache = {}

        return res

    @staticmethod
    def _get_domain_origin_key(domain, origin):
        origins_tuple = tuple((k, v) for k, v in sorted(origin.items()))
        return domain, origins_tuple

    def _sdfg_add_arrays_and_edges(
        self, wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs, origins
    ):
        device = gt_backend.from_name(self.backend).storage_info["device"]

        for name, array in inner_sdfg.arrays.items():
            if isinstance(array, dace.data.Array) and not array.transient:
                axes = self.field_info[name].axes

                shape = [f"__{name}_{axis}_size" for axis in axes] + [
                    d for d in self.field_info[name].data_dims
                ]

                wrapper_sdfg.add_array(
                    name,
                    dtype=array.dtype,
                    strides=array.strides,
                    shape=shape,
                    storage=dace.StorageType.GPU_Global
                    if device == "gpu"
                    else dace.StorageType.Default,
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
                        self.field_info[name].boundary.lower_indices,
                        inner_sdfg.arrays[name].shape,
                    )
                ]
                ranges += [(0, d, 1) for d in self.field_info[name].data_dims]
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

    def _sdfg_specialize_symbols(self, wrapper_sdfg, domain: Tuple[int, ...]):
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

            sdfg.replace("__I", ival)
            if "__I" in sdfg.symbols:
                sdfg.remove_symbol("__I")
            sdfg.replace("__J", jval)
            if "__J" in sdfg.symbols:
                sdfg.remove_symbol("__J")
            sdfg.replace("__K", kval)
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

    def _sdfg_freeze_domain_and_origin(
        self, inner_sdfg: dace.SDFG, domain: Tuple[int, ...], origin: Dict[str, Tuple[int, ...]]
    ):
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

        self._sdfg_add_arrays_and_edges(
            wrapper_sdfg, state, inner_sdfg, nsdfg, inputs, outputs, origins=origin
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

        self._sdfg_specialize_symbols(wrapper_sdfg, domain)

        for _, _, array in wrapper_sdfg.arrays_recursive():
            if array.transient:
                array.lifetime = dace.dtypes.AllocationLifetime.SDFG

        signature = self.__sdfg_signature__()
        wrapper_sdfg.arg_names = [a for a in signature[0] if a not in signature[1]]

        return wrapper_sdfg

    def freeze(
        self: "DaCeStencilObject", *, origin: Dict[str, Tuple[int, ...]], domain: Tuple[int, ...]
    ) -> DaCeFrozenStencil:

        key = DaCeStencilObject._get_domain_origin_key(domain, origin)
        if key in self._frozen_cache:
            return self._frozen_cache[key]

        frozen_hash = shash(origin, domain)

        # check if same sdfg already cached on disk
        basename = os.path.splitext(self.SDFG_PATH)[0]
        filename = basename + "_" + str(frozen_hash) + ".sdfg"
        try:
            frozen_sdfg = dace.SDFG.from_file(filename)
        except FileNotFoundError:
            # otherwise, wrap and save sdfg from scratch
            inner_sdfg = self.sdfg()

            frozen_sdfg = self._sdfg_freeze_domain_and_origin(inner_sdfg, domain, origin)
            frozen_sdfg.save(filename)

        self._frozen_cache[key] = DaCeFrozenStencil(self, origin, domain, frozen_sdfg)
        return self._frozen_cache[key]

    @classmethod
    def sdfg(cls) -> dace.SDFG:

        if getattr(cls, "_sdfg", None) is None:
            cls._sdfg = dace.SDFG.from_file(cls.SDFG_PATH)

        return copy.deepcopy(cls._sdfg)

    def closure_resolver(
        self,
        constant_args: Dict[str, Any],
        given_args: Set[str],
        parent_closure: Optional["dace.frontend.python.common.SDFGClosure"] = None,
    ) -> "dace.frontend.python.common.SDFGClosure":
        return dace.frontend.python.common.SDFGClosure()

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        norm_kwargs = self._normalize_args(*args, **kwargs)
        frozen_stencil = self.freeze(origin=norm_kwargs["origin"], domain=norm_kwargs["domain"])
        return frozen_stencil.__sdfg__(**norm_kwargs)

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {}

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        special_args = {"self", "domain", "origin", "validate_args", "exec_info"}
        args = []
        for arg in (
            inspect.getfullargspec(self.__call__).args
            + inspect.getfullargspec(self.__call__).kwonlyargs
        ):
            if arg in special_args:
                continue
            args.append(arg)
        return (args, [])

    def _normalize_args(self, *args, domain=None, origin=None, **kwargs):
        arg_names, consts = self.__sdfg_signature__()
        args_iter = iter(args)
        args_as_kwargs = {
            name: (kwargs[name] if name in kwargs else next(args_iter)) for name in arg_names
        }

        origin = self._normalize_origins(args_as_kwargs, origin)
        if domain is None:
            domain = self._get_max_domain(args_as_kwargs, origin)
        for key, value in kwargs.items():
            args_as_kwargs.setdefault(key, value)
        args_as_kwargs["domain"] = domain
        args_as_kwargs["origin"] = origin
        return args_as_kwargs
