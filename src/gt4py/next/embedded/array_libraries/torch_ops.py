# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
`ArrayOps` for torch tensors, and the embedded entry points under `torch.compile`.

Everything gt4py knows about `torch.compile` is in this module (ADR 0029).
An embedded program or field operator called in a region that `torch.compile`
captures runs as one `torch.compiler.nonstrict_trace` call: its fields and
connectivities are pytree nodes, and the embedded code runs on the fake tensors of
the trace. A connectivity table inside the trace is one of two cases:

- it stands for a concrete table, which it got through the pytree context: its
  static values are a NumPy array with the values of the concrete table;
- it was computed in the trace by `as_offset`: there are no static values, all
  its targets are assumed to be in range and the gather clamps them.

A connectivity reaches the trace through the pytree context only if it was
created by `common._connectivity`, e.g. `gtx.as_connectivity`, outside of the
region; `restrict`, `dataclasses.replace` and `NdArrayConnectivityField.from_array`
give connectivities that are not supported in the region.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, ClassVar, Optional

import array_api_compat.torch as compat_torch
import numpy as np
import torch
import torch.utils._pytree as torch_pytree

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import array_ops, nd_array_field
from gt4py.next.ffront import experimental, fbuiltins


def _cbrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)


def _gamma(x: torch.Tensor) -> torch.Tensor:
    sign = torch.where((x < 0) & (torch.floor(x) % 2 == 1), -1.0, 1.0).to(x.dtype)
    result = sign * torch.exp(torch.lgamma(x))
    # the poles as in SciPy: +-inf at +-0, NaN at the negative integers
    result = torch.where(x == 0, 1 / x, result)
    return torch.where((x < 0) & (x == torch.floor(x)), torch.nan, result)


def _is_concrete(tensor: Any) -> bool:
    # the tensors of a trace are subclasses: fake, functional or proxy tensors
    return type(tensor) is torch.Tensor


def _as_offset(offset: fbuiltins.FieldOffset, offset_field: nd_array_field.NdArrayField) -> Any:
    connectivity = nd_array_field._as_offset(offset, offset_field)
    assert isinstance(connectivity, nd_array_field.NdArrayConnectivityField)
    if not _is_concrete(connectivity.ndarray):
        # computed in the trace: its values exist only at run time
        return dataclasses.replace(
            connectivity, _static_values=nd_array_field.StaticValues.RUN_TIME_ONLY
        )
    return connectivity


class TorchOps(array_ops.ArrayOps):
    builtin_overrides: ClassVar[Mapping[Any, Callable[..., Any]]] = {
        experimental.as_offset: _as_offset
    }

    @property
    def array_ns(self) -> ModuleType:
        return torch

    def operand(self, value: Any, like: core_defs.NDArrayObject) -> Any:
        """
        `value` as an operand of an elementwise function next to the tensor `like`.

        Scalars become 0-d tensors, as several torch functions reject Python and NumPy
        scalars. They are created on the CPU: torch takes 0-d CPU tensors as operands of
        CUDA tensors, without a copy to the device. A Python scalar of the kind of `like`
        takes its dtype, to keep the weak typing of the standard.
        """
        if isinstance(value, bool | int | float):
            like_dtype = like.dtype
            like_kind = (
                bool
                if like_dtype == self.xp.bool
                else float
                if like_dtype.is_floating_point
                else int
            )
            dtype = like_dtype if type(value) is like_kind else self.dtype(type(value))
            return self.xp.asarray(value, dtype=dtype, device="cpu")
        if isinstance(value, np.generic):
            return self.xp.asarray(value, device="cpu")
        return value

    def _elementwise_function(self, name: str) -> Callable[..., Any]:
        if name == "cbrt":
            return _cbrt
        if name == "gamma":
            return _gamma
        return super()._elementwise_function(name)

    def to_numpy(self, array: core_defs.NDArrayObject) -> np.ndarray:
        return array.detach().cpu().numpy()  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete


array_ops.register(TorchOps(compat_torch))


class _TableIdentity:
    """
    The concrete table of a connectivity, for the pytree context: equal if it is the same memory.

    `torch.compile` guards a constant of the context by equality, so a call with another
    buffer, even with equal values, retraces, while a new connectivity around the same
    buffer does not. The values are read on first use, in a trace that needs them.
    """

    __slots__ = ("__weakref__", "_key", "_table", "_values")

    _key: tuple[int, tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]
    _table: torch.Tensor
    _values: Optional[np.ndarray]

    def __init__(self, table: torch.Tensor) -> None:
        assert _is_concrete(table)
        # holding the tensor keeps its memory alive, so no other tensor can get its address
        self._table = table
        self._key = (
            table.data_ptr(),
            tuple(table.shape),
            table.stride(),
            table.dtype,
            table.device,
        )
        self._values = None

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            # the trace computes on fake tensors, which have no values
            with torch._subclasses.fake_tensor.unset_fake_temporarily():
                values = self._table.detach().cpu().numpy()
            values.flags.writeable = False
            self._values = values
        return self._values

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TableIdentity) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)

    def __repr__(self) -> str:
        return f"<table of the tensor at {self._key[0]:#x}>"


torch_pytree.register_constant(_TableIdentity)

# the connectivity attribute with the identity of its concrete table, set by its creator
_TABLE_IDENTITY = "_torch_table_identity"


def _connectivity(data: torch.Tensor, /, *args: Any, **kwargs: Any) -> Any:
    connectivity = nd_array_field._connectivity_from_array(data, *args, **kwargs)
    if _is_concrete(connectivity.ndarray):
        connectivity.__dict__[_TABLE_IDENTITY] = _TableIdentity(connectivity.ndarray)
    return connectivity


common._connectivity.register(torch.Tensor, _connectivity)


# pytree contexts are plain constants: objects created in the region, as the domains of
# the fields a call returns, are no constants for `torch.compile`
_DomainContext = tuple[tuple[str, common.DimensionKind, int, int], ...]


def _domain_context(domain: common.Domain) -> _DomainContext:
    return tuple(
        (dim.value, dim.kind, rng.start, rng.stop)
        for dim, rng in zip(domain.dims, domain.ranges, strict=True)
    )


def _context_domain(context: _DomainContext) -> common.Domain:
    return common.Domain(
        dims=tuple(common.Dimension(value, kind) for value, kind, _, _ in context),
        ranges=tuple(common.UnitRange(start, stop) for _, _, start, stop in context),
    )


def _check_tensor(kind: str, ndarray: Any) -> None:
    if not isinstance(ndarray, torch.Tensor):
        raise TypeError(f"Only {kind} of torch tensors are torch pytrees, got '{type(ndarray)}'.")


def _flatten_field(field: nd_array_field.NdArrayField) -> tuple[list[torch.Tensor], _DomainContext]:
    _check_tensor("fields", field.ndarray)
    return [field.ndarray], _domain_context(field.domain)


def _unflatten_field(
    children: list[torch.Tensor], context: _DomainContext
) -> nd_array_field.NdArrayField:
    return nd_array_field.NdArrayField(_context_domain(context), children[0])


torch_pytree.register_pytree_node(nd_array_field.NdArrayField, _flatten_field, _unflatten_field)


_ConnectivityContext = tuple[
    _DomainContext, str, common.DimensionKind, Optional[int], Optional[_TableIdentity]
]


def _flatten_connectivity(
    connectivity: nd_array_field.NdArrayConnectivityField,
) -> tuple[list[torch.Tensor], _ConnectivityContext]:
    ndarray = connectivity.ndarray
    _check_tensor("connectivities", ndarray)
    identity = getattr(connectivity, _TABLE_IDENTITY, None)
    if (
        identity is None
        and connectivity._static_values is not nd_array_field.StaticValues.RUN_TIME_ONLY
    ):
        if torch.compiler.is_compiling() or not _is_concrete(ndarray):
            raise TypeError(
                "A connectivity in a 'torch.compile' region has to be created by"
                " 'gtx.as_connectivity' outside of the region."
            )
        identity = _TableIdentity(ndarray)
    codomain = connectivity.codomain
    skip_value = connectivity.skip_value
    return [ndarray], (
        _domain_context(connectivity.domain),
        codomain.value,
        codomain.kind,
        None if skip_value is None else int(skip_value),
        identity,
    )


def _unflatten_connectivity(
    children: list[torch.Tensor], context: _ConnectivityContext
) -> nd_array_field.NdArrayConnectivityField:
    (ndarray,) = children
    domain, codomain_value, codomain_kind, skip_value, identity = context
    static_values: Any
    if _is_concrete(ndarray):
        static_values = nd_array_field.StaticValues.IN_BUFFER
    elif identity is None or tuple(ndarray.shape) != identity.values.shape:
        static_values = nd_array_field.StaticValues.RUN_TIME_ONLY
    else:
        static_values = identity.values
    connectivity: nd_array_field.NdArrayConnectivityField = nd_array_field.NdArrayConnectivityField(
        _context_domain(domain),
        ndarray,
        common.Dimension(codomain_value, codomain_kind),
        skip_value,
        _static_values=static_values,
    )  # type: ignore[abstract, unused-ignore] # see 'jax_ops._unflatten_connectivity'
    if identity is not None:
        # a trace flattens the connectivities it unflattened again
        connectivity.__dict__[_TABLE_IDENTITY] = identity
    return connectivity


# pytree nodes are registered by exact type, so the connectivity needs its own registration
torch_pytree.register_pytree_node(
    nd_array_field.NdArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see 'jax_ops'
    _flatten_connectivity,
    _unflatten_connectivity,
)


# -- embedded entry points under `torch.compile` --

_entry_points: weakref.WeakValueDictionary[int, Callable[..., Any]] = weakref.WeakValueDictionary()
_within_nonstrict_call = False


def register_entry_point(entry_point: Callable[..., Any]) -> None:
    """Make `entry_point` callable in a `torch.compile` region; before the region, which applies mutations only after tracing."""
    _entry_points[id(entry_point)] = entry_point


def is_compiling() -> bool:
    """Whether an embedded entry point is called in a `torch.compile` region, outside of `call`."""
    return not _within_nonstrict_call and torch.compiler.is_compiling()


def call(entry_point: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
    """Call `entry_point` in a `torch.compile` region as one non-strictly traced function."""
    if "domain" in kwargs:
        kwargs = {**kwargs, "domain": _encode_domain_argument(kwargs["domain"])}
    result = _nonstrict_call(id(entry_point), args, kwargs)
    return result[0] if result else None


def _encode_domain_argument(domain: Any) -> Any:
    # the dimensions of a domain, e.g. as dict keys, cannot be in a pytree context; a list is
    # no domain of a tuple output
    if isinstance(domain, tuple):
        return tuple(_encode_domain_argument(d) for d in domain)
    return list(_domain_context(common.domain(domain)))


def _decode_domain_argument(domain: Any) -> Any:
    if isinstance(domain, tuple):
        return tuple(_decode_domain_argument(d) for d in domain)
    return _context_domain(tuple(domain))


@torch.compiler.nonstrict_trace
def _nonstrict_call(entry_point_id: int, args: tuple, kwargs: dict[str, Any]) -> tuple[Any, ...]:
    global _within_nonstrict_call
    _within_nonstrict_call = True
    try:
        if "domain" in kwargs:
            kwargs = {**kwargs, "domain": _decode_domain_argument(kwargs["domain"])}
        result = _entry_points[entry_point_id](*args, **kwargs)
    finally:
        _within_nonstrict_call = False
    # a `None` output is not supported by `nonstrict_trace`
    return () if result is None else (result,)
