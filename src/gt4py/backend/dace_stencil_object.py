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

from gt4py.definitions import AccessKind
from gt4py.stencil_object import FrozenStencil, StencilObject
from gt4py.utils import shash


def add_optional_fields(sdfg, field_info, parameter_info, **kwargs):
    for name, info in field_info.items():
        if info.access == AccessKind.NONE and name in kwargs:
            outer_array = kwargs[name]
            sdfg.add_array(
                name,
                shape=outer_array.shape,
                dtype=outer_array.dtype,
                strides=outer_array.strides,
            )

    for name, info in parameter_info.items():
        if info.access == AccessKind.NONE and name in kwargs:
            if isinstance(kwargs[name], dace.data.Scalar):
                sdfg.add_symbol(name, stype=kwargs[name].dtype)
            else:
                sdfg.add_symbol(name, stype=dace.typeclass(type(kwargs[name])))
    return sdfg


@dataclass(frozen=True)
class DaCeFrozenStencil(FrozenStencil, SDFGConvertible):

    stencil_object: "DaCeStencilObject"
    origin: Dict[str, Tuple[int, ...]]
    domain: Tuple[int, ...]
    sdfg: dace.SDFG

    def __sdfg__(self, **kwargs):
        return add_optional_fields(
            self.sdfg, self.stencil_object.field_info, self.stencil_object.parameter_info, **kwargs
        )

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
            from gt4py.backend.dace_backend import freeze_origin_domain_sdfg

            frozen_sdfg = freeze_origin_domain_sdfg(
                inner_sdfg,
                arg_names=self.__sdfg_signature__()[0],
                field_info=self.field_info,
                origin=origin,
                domain=domain,
            )
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
        arg_names, _ = self.__sdfg_signature__()
        norm_kwargs = DaCeStencilObject.normalize_args(
            *args,
            arg_names=arg_names,
            domain_info=self.domain_info,
            field_info=self.field_info,
            **kwargs,
        )
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

    @staticmethod
    def normalize_args(
        *args, arg_names, domain_info, field_info, domain=None, origin=None, **kwargs
    ):
        args_iter = iter(args)
        args_as_kwargs = {
            name: (kwargs[name] if name in kwargs else next(args_iter)) for name in arg_names
        }

        origin = DaCeStencilObject._normalize_origins(args_as_kwargs, field_info, origin)
        if domain is None:
            domain = DaCeStencilObject._get_max_domain(
                args_as_kwargs, domain_info, field_info, origin
            )
        for key, value in kwargs.items():
            args_as_kwargs.setdefault(key, value)
        args_as_kwargs["domain"] = domain
        args_as_kwargs["origin"] = origin
        return args_as_kwargs
