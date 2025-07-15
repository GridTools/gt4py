# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple

import dace
import dace.data
import dace.frontend.python.common
from dace.frontend.python.common import SDFGClosure, SDFGConvertible

from gt4py.cartesian import backend as gt_backend
from gt4py.cartesian.backend.dace_backend import freeze_origin_domain_sdfg
from gt4py.cartesian.definitions import AccessKind, DomainInfo, FieldInfo
from gt4py.cartesian.stencil_object import ArgsInfo, FrozenStencil, StencilObject
from gt4py.cartesian.utils import shash


def _extract_array_infos(field_args, device) -> Dict[str, Optional[ArgsInfo]]:
    return {
        name: ArgsInfo(
            array=arg,
            dimensions=getattr(arg, "__gt_dims__", None),
            device=device,
            origin=getattr(arg, "__gt_origin__", None),
        )
        for name, arg in field_args.items()
    }


def add_optional_fields(
    sdfg: dace.SDFG,
    field_info: Dict[str, Any],
    parameter_info: Dict[str, Any],
    **kwargs: Any,
) -> dace.SDFG:
    sdfg = copy.deepcopy(sdfg)
    for name, info in field_info.items():
        if info.access == AccessKind.NONE and name in kwargs and name not in sdfg.arrays:
            outer_array = kwargs[name]
            sdfg.add_array(
                name,
                shape=outer_array.shape,
                dtype=outer_array.dtype,
                strides=outer_array.strides,
            )

    for name, info in parameter_info.items():
        if info.access == AccessKind.NONE and name in kwargs and name not in sdfg.symbols:
            if isinstance(kwargs[name], dace.data.Scalar):
                sdfg.add_scalar(name, dtype=kwargs[name].dtype)
    return sdfg


@dataclass(frozen=True)
class DaCeFrozenStencil(FrozenStencil, SDFGConvertible):
    stencil_object: DaCeStencilObject
    origin: Dict[str, Tuple[int, ...]]
    domain: Tuple[int, ...]
    sdfg: dace.SDFG

    def __sdfg__(self, **kwargs):
        return add_optional_fields(
            self.sdfg,
            self.stencil_object.field_info,
            self.stencil_object.parameter_info,
            **kwargs,
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
        self: DaCeStencilObject,
        *,
        origin: Dict[str, Tuple[int, ...]],
        domain: Tuple[int, ...],
    ) -> DaCeFrozenStencil:
        key = DaCeStencilObject._get_domain_origin_key(domain, origin)

        # check if same sdfg already cached on disk
        if key in self._frozen_cache:
            return self._frozen_cache[key]

        # otherwise, wrap and save sdfg from scratch
        backend_class = gt_backend.from_name(self.backend)
        frozen_sdfg = freeze_origin_domain_sdfg(
            self.sdfg(),
            arg_names=list(self.__sdfg_signature__()[0]),
            field_info=self.field_info,
            layout_info=backend_class.storage_info,
            origin=origin,
            domain=domain,
        )
        self._frozen_cache[key] = DaCeFrozenStencil(self, origin, domain, frozen_sdfg)

        basename = os.path.splitext(self.SDFG_PATH)[0]
        filename = f"{basename}_{shash(origin, domain)}.sdfg"
        frozen_sdfg.save(filename)

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
        parent_closure: Optional[dace.frontend.python.common.SDFGClosure] = None,
    ) -> dace.frontend.python.common.SDFGClosure:
        return dace.frontend.python.common.SDFGClosure()

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        arg_names, _ = self.__sdfg_signature__()
        norm_kwargs = DaCeStencilObject.normalize_arg_fields(
            *args,
            backend=self.backend,
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
    def normalize_arg_fields(
        *args,
        backend: str,
        arg_names: Iterable[str],
        domain_info: DomainInfo,
        field_info: Dict[str, FieldInfo],
        domain: Optional[Tuple[int, ...]] = None,
        origin: Optional[Dict[str, Tuple[int, ...]]] = None,
        **kwargs,
    ):
        """Normalize Fields in argument list to the proper domain/origin"""
        backend_cls = gt_backend.from_name(backend)
        args_iter = iter(args)

        args_as_kwargs = {}
        for name in arg_names:
            if name not in field_info.keys():
                continue
            if name in kwargs.keys():
                args_as_kwargs[name] = kwargs[name]
            else:
                args_as_kwargs[name] = next(args_iter)

        arg_infos = _extract_array_infos(
            field_args=args_as_kwargs, device=backend_cls.storage_info["device"]
        )

        origin = DaCeStencilObject._normalize_origins(arg_infos, field_info, origin)

        if domain is None:
            domain = DaCeStencilObject._get_max_domain(arg_infos, domain_info, field_info, origin)
        for key, value in kwargs.items():
            args_as_kwargs.setdefault(key, value)
        args_as_kwargs["domain"] = domain
        args_as_kwargs["origin"] = origin
        return args_as_kwargs
